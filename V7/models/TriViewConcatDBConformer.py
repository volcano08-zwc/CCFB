"""Tri-view DBConformer with information-preserving weighted concatenation.

This variant keeps the temporal, spatial, frequency, bottleneck-interaction,
and feedback paths from ``TriViewDBConformer``.  It changes only the final
adaptive fusion:

    weighted sum:     w_T F_T + w_S F_S + w_F F_F       -> D
    weighted concat: [w_T F_T ; w_S F_S ; w_F F_F]      -> 3D

For the default D=40, the classifier therefore receives 120 features.
"""

import torch
from torch import Tensor, nn

from config import TRIVIEW_CONFIG
from models.DBConformer import ClassificationHead
from models.TriViewDBConformer import TriViewDBConformer


class WeightedConcatAdaptiveFusion(nn.Module):
    """Learn T/S/F view weights while preserving each view's feature slots."""

    def __init__(self, emb_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.view_gate = nn.Sequential(
            nn.Linear(emb_size * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        # Begin with equal view weights.  Training can then specialize the
        # weights per trial without imposing an arbitrary initial preference.
        nn.init.zeros_(self.view_gate[-1].weight)
        nn.init.zeros_(self.view_gate[-1].bias)

    def forward(
        self,
        temporal: Tensor,
        spatial: Tensor,
        frequency: Tensor,
    ):
        views_concat = torch.cat([temporal, spatial, frequency], dim=-1)
        weights = torch.softmax(self.view_gate(views_concat), dim=-1)

        weighted_temporal = weights[:, 0:1] * temporal
        weighted_spatial = weights[:, 1:2] * spatial
        weighted_frequency = weights[:, 2:3] * frequency
        fused = torch.cat(
            [weighted_temporal, weighted_spatial, weighted_frequency], dim=-1
        )
        return fused, weights


class TriViewConcatDBConformer(TriViewDBConformer):
    """120-D weighted-concatenation variant of TriViewDBConformer."""

    def __init__(
        self,
        args,
        emb_size=None,
        tem_depth=None,
        chn_depth=None,
        chn=None,
        n_classes=None,
    ):
        super().__init__(
            args,
            emb_size=emb_size,
            tem_depth=tem_depth,
            chn_depth=chn_depth,
            chn=chn,
            n_classes=n_classes,
        )

        config = TRIVIEW_CONFIG
        emb_size = (
            emb_size
            if emb_size is not None
            else getattr(args, "emb_size", config.emb_size)
        )
        n_classes = n_classes if n_classes is not None else args.class_num

        self.adaptive_fusion = WeightedConcatAdaptiveFusion(
            emb_size=emb_size,
            hidden_dim=getattr(args, "fusion_hidden_dim", config.fusion_hidden_dim),
            dropout=getattr(args, "fusion_dropout", config.fusion_dropout),
        )
        self.classifier = ClassificationHead(emb_size * 3, n_classes)


def build_triview_concat_dbconformer(args):
    """Factory matching the argument object used by DBConformer_LOSO.py."""
    return TriViewConcatDBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )

