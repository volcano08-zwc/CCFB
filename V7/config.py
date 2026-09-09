"""Architecture defaults for the tri-view DBConformer variant.

The LOSO script keeps ownership of training/data hyperparameters.  This file
contains only model-specific defaults so the new branch can be tuned without
touching data loading, Euclidean Alignment, or the training loop.
"""

from dataclasses import dataclass
from typing import Tuple


FrequencyBand = Tuple[float, float]


@dataclass(frozen=True)
class TriViewModelConfig:
    # Original DBConformer defaults (dataset-specific args still take priority).
    emb_size: int = 40
    spa_dim: int = 16
    temporal_depth: int = 2
    spatial_depth: int = 2

    # Lightweight frequency branch.  The 8--13 Hz mu band and the split beta
    # bands (13--20/20--30 Hz) are explicitly represented.
    frequency_bands: Tuple[FrequencyBand, ...] = (
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 20.0),
        (20.0, 30.0),
        (30.0, 40.0),
    )
    frequency_filter_kernel: int = 63
    spectral_kernel: int = 7
    frequency_dropout: float = 0.20

    # Three-token bottleneck interaction.
    interaction_heads: int = 4
    interaction_dropout: float = 0.20
    interaction_ff_expansion: int = 2
    feedback_gate_init: float = 0.05

    # Small view-weighting head used by AdaptiveFusion.
    fusion_hidden_dim: int = 16
    fusion_dropout: float = 0.10


TRIVIEW_CONFIG = TriViewModelConfig()

