"""Structural neural-GARCH: econometric recurrences inside the network.

This is the package's identity. The models here do not use GARCH as a feature
generator; the recursion is part of the computation graph and is trained jointly
with the neural parameters.
"""

from .embedded import (
    EmbeddedGARCHGRU, EmbeddedGARCHGRUCell, EmbeddedGARCHLSTM, EmbeddedGARCHLSTMCell,
)
from .garch_nn import GARCHNN, StructuralGARCHLSTM
from .neural_garch import NeuralBEKK, NeuralGARCH
from .realrech import DeepRGARCH, RealizedGARCH, RealRECH
from .amortized import (
    AmortizedGARCHInference, GARCHParameters, NeuralGARCHEstimator,
    empirical_moments, fourth_standardised_moment, garch_moment_map,
    normalised_autocovariance,
)
from .garchnet import ANNGARCH, GARCHNet, LikelihoodGARCHLSTM
from .rech import LSTMtGARCH, RECH, SRNGARCH, SRNGJR, bounded_relu
from .sigma_cell import (
    SigmaCell, SigmaCellN, SigmaCellNTV, SigmaCellRL, SigmaCellRLTV,
    SigmaLSTM, adjusted_softplus, sigma_cell_loss,
)
from .kernels import (
    FIGARCHNNKernel, GARCHKernel, GARCHNNKernel, GJRNNKernel, figarch_weights, get_kernel,
)

__all__ = [
    "GARCHKernel", "GARCHNNKernel", "GJRNNKernel", "FIGARCHNNKernel",
    "get_kernel", "figarch_weights",
    "GARCHNN", "StructuralGARCHLSTM", "NeuralGARCH", "NeuralBEKK",
    "RECH", "SRNGARCH", "SRNGJR", "LSTMtGARCH", "bounded_relu",
    "RealizedGARCH", "DeepRGARCH", "RealRECH",
    "SigmaCell", "SigmaCellN", "SigmaCellRL", "SigmaCellNTV", "SigmaCellRLTV",
    "SigmaLSTM", "adjusted_softplus", "sigma_cell_loss",
    "GARCHNet", "ANNGARCH", "LikelihoodGARCHLSTM",
    "AmortizedGARCHInference", "NeuralGARCHEstimator", "GARCHParameters",
    "garch_moment_map", "normalised_autocovariance",
    "fourth_standardised_moment", "empirical_moments",
    "EmbeddedGARCHGRUCell", "EmbeddedGARCHLSTMCell",
    "EmbeddedGARCHGRU", "EmbeddedGARCHLSTM",
]
