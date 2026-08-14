"""GARCH-informed learning: econometric structure enters the loss, not the graph.

The third integration route in the taxonomy. Zhao et al. put GARCH in the
architecture, Wei et al. in the recurrent gating; here it appears only in the
objective, as a regulariser pulling the network toward an econometric forecast.
"""

from .ginn import GINN, GINN0, GARCHInformedNetwork, GINNDataset, build_ginn_inputs

__all__ = ["GINN", "GINN0", "GINNDataset", "GARCHInformedNetwork", "build_ginn_inputs"]
