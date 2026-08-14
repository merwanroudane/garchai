"""Feature hybrids and ensembles — the baseline family.

GARCH as an input to a learner, rather than a component of one. These are
benchmarks: `hybridecon` owns the plain GARCH-LSTM/GRU variants, and what is
built here is only what it does not cover — residual correction, stacking, and
transformer fusion.
"""

from .garch_tft import (
    GARCHTFT, GatedLinearUnit, GatedResidualNetwork,
    InterpretableMultiHeadAttention, VariableSelectionNetwork,
    garman_klass, historical_volatility, quantile_loss,
)
from .sentiment import (
    SentimentGARCHLSTM, SentimentGARCHX, normalise_sentiment, vix_sentiment_proxy,
)
from .hybrids import (
    GARCHFeatures, GARCHXGBoost, LSTMANNGARCH, MultiGARCHTransformer,
    StackedMLGARCH, build_garch_features,
)

__all__ = [
    "GARCHFeatures", "build_garch_features", "LSTMANNGARCH",
    "GARCHXGBoost", "StackedMLGARCH", "MultiGARCHTransformer",
    "GARCHTFT", "GatedLinearUnit", "GatedResidualNetwork",
    "InterpretableMultiHeadAttention", "VariableSelectionNetwork",
    "historical_volatility", "garman_klass", "quantile_loss",
    "SentimentGARCHX", "SentimentGARCHLSTM", "normalise_sentiment",
    "vix_sentiment_proxy",
]
