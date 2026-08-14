"""Graph and multi-scale volatility models.

GENSHIN is the one model in the collection whose contribution is mostly in
*feature construction* rather than architecture: 60 GARCH fits, decomposed by
frequency band and compressed against the target, before any network sees them.
"""

from .genshin import (
    DISTRIBUTIONS, GARCHFeatureBank, GENSHIN, GENSHINFeatures, LAG_STRUCTURES,
    MSGNet, SPECIFICATIONS,
)
from .vmd import variational_mode_decomposition, vmd_reconstruct

__all__ = [
    "variational_mode_decomposition", "vmd_reconstruct",
    "GENSHIN", "GENSHINFeatures", "GARCHFeatureBank", "MSGNet",
    "SPECIFICATIONS", "LAG_STRUCTURES", "DISTRIBUTIONS",
]
