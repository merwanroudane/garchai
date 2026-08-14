"""garchai - econometrically constrained, differentiable, AI-enhanced
conditional heteroskedasticity models.

The package is organised around two primitives: a differentiable
:class:`~garchai.cells.base.VarianceCell` carrying a GARCH-family recursion
inside the autograd graph, and a family-specific constraint transform that keeps
the parameters econometrically valid at every time step.

Scope note
----------
``garchai`` deliberately does not duplicate ``hybridecon``. Feature hybrids
(GARCH volatility as an input feature to a neural network) live there and are
reached from here through :mod:`garchai.adapters`. ``garchai`` owns the
structural, embedded, informed, Bayesian and constrained-multivariate families.
"""

__version__ = "0.1.0.dev0"
__author__ = "Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__url__ = "https://github.com/merwanroudane/garchai"

from . import (
    cells, constraints, core, datasets, distributions, econometric,
    evaluation, graph, informed, losses, mean, reporting, risk, structural, targets,
)
from .core import ModelConfig, NumericalConfig, Provenance, TrainConfig, set_seed
from .econometric import EGARCH, GARCH, GJRGARCH
from .structural import (
    RECH, AmortizedGARCHInference, ANNGARCH, DeepRGARCH, GARCHNet,
    SigmaCell, SigmaLSTM, EmbeddedGARCHGRU, EmbeddedGARCHLSTM, GARCHNN, LSTMtGARCH,
    NeuralBEKK, NeuralGARCH, RealizedGARCH, RealRECH, SRNGARCH, SRNGJR,
    StructuralGARCHLSTM,
)

__all__ = [
    "__version__", "__author__", "__email__", "__url__",
    "core", "cells", "constraints", "distributions", "econometric",
    "losses", "mean", "risk", "targets", "datasets", "evaluation", "reporting", "structural",
    "ModelConfig", "NumericalConfig", "TrainConfig", "Provenance", "set_seed",
    "GARCH", "GJRGARCH", "EGARCH",
    "GARCHNN", "StructuralGARCHLSTM", "EmbeddedGARCHGRU", "EmbeddedGARCHLSTM",
    "NeuralGARCH", "NeuralBEKK",
    "RECH", "SRNGARCH", "SRNGJR", "LSTMtGARCH",
    "RealizedGARCH", "DeepRGARCH", "RealRECH",
    "SigmaCell", "SigmaLSTM", "GARCHNet", "ANNGARCH",
    "AmortizedGARCHInference",
]
