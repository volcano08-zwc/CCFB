"""Minimal tri-view extension of the original DBConformer.

Temporal/spatial embeddings, Conformer blocks, channel attention, and the MLP
classifier follow the original implementation.  The only additions are a
lightweight frequency branch, a three-token interaction bottleneck, residual
feedback, and a small adaptive fusion head.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import TRIVIEW_CONFIG
from models.DBConformer import (
    ClassificationHead,
    FeedForwardBlock,
    MultiHeadAttention,
    PatchEmbeddingSpatial,
    PatchEmbeddingTemporal,
    TransformerEncoder,
    TransformerEncoderBlock,
)


class LearnableFilterBank(nn.Module):
    """Shared learnable band-pass filters applied independently per channel."""

    def __init__(
        self,
        sample_rate: float,
        bands: Sequence[Tuple[float, float]],
        kernel_size: int,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("frequency_filter_kernel must be odd")
        if not bands:
            raise ValueError("frequency_bands must contain at least one band")

        nyquist = sample_rate / 2.0
        for low, high in bands:
            if not 0.0 <= low < high < nyquist:
                raise ValueError(
                    f"Invalid frequency band ({low}, {high}) for "
                    f"sample_rate={sample_rate}; require 0 <= low < high < {nyquist}."
                )

        self.num_bands = len(bands)
        self.filters = nn.Conv1d(
            1,
            self.num_bands,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self._init_as_bandpass(sample_rate, bands, kernel_size)

    def _init_as_bandpass(self, sample_rate, bands, kernel_size):
        """Initialize learnable kernels with windowed-sinc band-pass filters."""
        center = (kernel_size - 1) / 2.0
        n = torch.arange(kernel_size, dtype=torch.float32) - center
        window = torch.hamming_window(kernel_size, periodic=False)
        kernels = []
        for low, high in bands:
            low_norm = low / sample_rate
            high_norm = high / sample_rate
            high_pass = 2.0 * high_norm * torch.sinc(2.0 * high_norm * n)
            low_pass = 2.0 * low_norm * torch.sinc(2.0 * low_norm * n)
            kernel = (high_pass - low_pass) * window
            kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
            kernels.append(kernel)
        with torch.no_grad():
            self.filters.weight.copy_(torch.stack(kernels).unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T); filters are shared across electrodes.
        batch, channels, time_points = x.shape
        x = x.reshape(batch * channels, 1, time_points)
        x = self.filters(x)
        return x.view(batch, channels, self.num_bands, time_points)


class FrequencyAttentionPooling(nn.Module):
    """Convert filtered signals into band tokens and pool across frequency."""

    def __init__(self, num_bands: int, emb_size: int, dropout: float):
        super().__init__()
        self.band_projection = nn.Linear(1, emb_size)
        self.band_embedding = nn.Parameter(torch.zeros(1, num_bands, emb_size))
        self.attention = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.band_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Stable log-power summary over channels and time: (B, num_bands).
        band_power = torch.log1p(x.square().mean(dim=(1, 3)))
        band_tokens = self.band_projection(band_power.unsqueeze(-1))
        band_tokens = self.dropout(band_tokens + self.band_embedding)
        weights = torch.softmax(self.attention(band_tokens), dim=1)
        summary = torch.sum(weights * band_tokens, dim=1)
        return summary, weights


class LightweightFrequencyBranch(nn.Module):
    """Filter bank + shared depthwise spectral convolution + attention pooling."""

    def __init__(
        self,
        sample_rate: float,
        bands: Sequence[Tuple[float, float]],
        filter_kernel: int,
        spectral_kernel: int,
        emb_size: int,
        dropout: float,
    ):
        super().__init__()
        if spectral_kernel % 2 == 0:
            raise ValueError("spectral_kernel must be odd")
        self.filter_bank = LearnableFilterBank(sample_rate, bands, filter_kernel)
        num_bands = len(bands)
        # Applied to (B*C, F, T), so parameters are independent of C and the
        # convolution is depthwise over the learned frequency-band signals.
        self.spectral_conv = nn.Sequential(
            nn.Conv1d(
                num_bands,
                num_bands,
                kernel_size=spectral_kernel,
                padding=spectral_kernel // 2,
                groups=num_bands,
                bias=False,
            ),
            nn.BatchNorm1d(num_bands),
            nn.GELU(),
        )
        # Start as an identity refinement so the initialized filter-bank bands
        # remain meaningful before this depthwise layer learns a better shape.
        spectral_weight = self.spectral_conv[0].weight
        with torch.no_grad():
            spectral_weight.zero_()
            spectral_weight[:, 0, spectral_kernel // 2] = 1.0
        self.pool = FrequencyAttentionPooling(num_bands, emb_size, dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        filtered = self.filter_bank(x)  # (B, C, F, T)
        batch, channels, bands, time_points = filtered.shape
        filtered = filtered.reshape(batch * channels, bands, time_points)
        filtered = self.spectral_conv(filtered)
        filtered = filtered.view(batch, channels, bands, time_points)
        return self.pool(filtered)


class TriViewBottleneckInteraction(nn.Module):
    """One Transformer-style interaction over exactly three view tokens."""

    def __init__(self, emb_size, num_heads, dropout, ff_expansion):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by interaction_heads")
        self.norm_attention = nn.LayerNorm(emb_size)
        self.attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, ff_expansion, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(emb_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention_dropout(self.attention(self.norm_attention(x)))
        x = x + self.ffn_dropout(self.ffn(self.norm_ffn(x)))
        return self.output_norm(x)


class AdaptiveFusion(nn.Module):
    """Learn sample-wise T/S/F weights from their concatenated summaries."""

    def __init__(self, emb_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.view_gate = nn.Sequential(
            nn.Linear(emb_size * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        # Equal T/S/F weights at initialization; learning then specializes the
        # weights per sample rather than beginning with an arbitrary preference.
        nn.init.zeros_(self.view_gate[-1].weight)
        nn.init.zeros_(self.view_gate[-1].bias)

    def forward(self, temporal: Tensor, spatial: Tensor, frequency: Tensor):
        concatenated = torch.cat([temporal, spatial, frequency], dim=-1)
        weights = torch.softmax(self.view_gate(concatenated), dim=-1)
        views = torch.stack([temporal, spatial, frequency], dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * views, dim=1)
        return fused, weights


class TriViewDBConformer(nn.Module):
    """DBConformer with lightweight frequency-guided tri-view interaction."""

    def __init__(self, args, emb_size=None, tem_depth=None, chn_depth=None,
                 chn=None, n_classes=None):
        super().__init__()
        config = TRIVIEW_CONFIG
        emb_size = emb_size if emb_size is not None else getattr(args, "emb_size", config.emb_size)
        tem_depth = tem_depth if tem_depth is not None else getattr(
            args, "transformer_depth_tem", config.temporal_depth)
        chn_depth = chn_depth if chn_depth is not None else getattr(
            args, "transformer_depth_chn", config.spatial_depth)
        n_classes = n_classes if n_classes is not None else args.class_num
        if tem_depth < 2 or chn_depth < 2:
            raise ValueError("TriViewDBConformer requires at least two T/S blocks")

        self.embedding = PatchEmbeddingTemporal(
            data_name=args.data_name,
            in_planes=args.chn,
            out_planes=emb_size,
            kernel_size=63,
            radix=1,
            patch_size=args.patch_size,
            time_points=args.time_sample_num,
            num_classes=args.class_num,
        )
        self.channel_embedding = PatchEmbeddingSpatial(
            spa_dim=getattr(args, "spa_dim", config.spa_dim),
            emb_size=emb_size,
        )

        self.P = args.time_sample_num // args.patch_size
        self.C = args.chn
        self.D = emb_size
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, self.P, self.D))
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, self.C, self.D))

        # Preserve the original total depth: interaction follows block 1 and
        # the remaining original blocks form stage 2 (one block for depth=2).
        self.temporal_block1 = TransformerEncoderBlock(emb_size)
        self.spatial_block1 = TransformerEncoderBlock(emb_size)
        self.temporal_block2 = TransformerEncoder(tem_depth - 1, emb_size)
        self.spatial_block2 = TransformerEncoder(chn_depth - 1, emb_size)

        bands = getattr(args, "frequency_bands", config.frequency_bands)
        self.frequency_branch = LightweightFrequencyBranch(
            sample_rate=args.sample_rate,
            bands=bands,
            filter_kernel=getattr(
                args, "frequency_filter_kernel", config.frequency_filter_kernel),
            spectral_kernel=getattr(args, "spectral_kernel", config.spectral_kernel),
            emb_size=emb_size,
            dropout=getattr(args, "frequency_dropout", config.frequency_dropout),
        )

        self.spatial_attn_pool = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, 1),
        )
        self.tri_view_interaction = TriViewBottleneckInteraction(
            emb_size=emb_size,
            num_heads=getattr(args, "interaction_heads", config.interaction_heads),
            dropout=getattr(
                args, "interaction_dropout", config.interaction_dropout),
            ff_expansion=getattr(
                args, "interaction_ff_expansion", config.interaction_ff_expansion),
        )

        gate_init = getattr(args, "feedback_gate_init", config.feedback_gate_init)
        if not 0.0 < gate_init < 1.0:
            raise ValueError("feedback_gate_init must be in (0, 1)")
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.feedback_gate_logits = nn.Parameter(torch.full((3,), gate_logit))

        self.adaptive_fusion = AdaptiveFusion(
            emb_size=emb_size,
            hidden_dim=getattr(args, "fusion_hidden_dim", config.fusion_hidden_dim),
            dropout=getattr(args, "fusion_dropout", config.fusion_dropout),
        )
        # AdaptiveFusion returns D dimensions, so the original MLP topology is
        # retained while its input size changes from 2D to D.
        self.classifier = ClassificationHead(emb_size, n_classes)

    def _spatial_pool(self, x: Tensor) -> Tensor:
        scores = self.spatial_attn_pool(x)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights * x, dim=1)

    def forward(self, x: Tensor):
        x = x.squeeze(1)  # (B, C, T), same external input contract as original

        temporal_tokens = self.embedding(x) + self.pos_embedding_temporal
        spatial_tokens = self.channel_embedding(x) + self.pos_embedding_spatial

        temporal_tokens = self.temporal_block1(temporal_tokens)
        spatial_tokens = self.spatial_block1(spatial_tokens)

        z_temporal = temporal_tokens.mean(dim=1)
        z_spatial = self._spatial_pool(spatial_tokens)
        z_frequency, _ = self.frequency_branch(x)

        tri_tokens = torch.stack([z_temporal, z_spatial, z_frequency], dim=1)
        interacted = self.tri_view_interaction(tri_tokens)
        z_temporal_prime, z_spatial_prime, z_frequency_prime = interacted.unbind(dim=1)

        gates = torch.sigmoid(self.feedback_gate_logits)
        temporal_tokens = temporal_tokens + gates[0] * z_temporal_prime.unsqueeze(1)
        spatial_tokens = spatial_tokens + gates[1] * z_spatial_prime.unsqueeze(1)
        frequency_feature = z_frequency + gates[2] * z_frequency_prime

        temporal_tokens = self.temporal_block2(temporal_tokens)
        spatial_tokens = self.spatial_block2(spatial_tokens)
        temporal_feature = temporal_tokens.mean(dim=1)
        spatial_feature = self._spatial_pool(spatial_tokens)

        fused, _ = self.adaptive_fusion(
            temporal_feature, spatial_feature, frequency_feature)
        _, logits = self.classifier(fused)
        return fused, logits


def build_triview_dbconformer(args):
    """Factory matching the argument object used by DBConformer_LOSO.py."""
    return TriViewDBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )
