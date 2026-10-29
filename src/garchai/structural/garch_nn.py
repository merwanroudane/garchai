r"""Zhao et al. (AAAI 2024): GARCH-NN and Structural GARCH-LSTM.

Two models, both built on the kernels in :mod:`garchai.structural.kernels`:

:class:`GARCHNN`
    The bare NN counterpart of a GARCH-family model, trained by
    backpropagation instead of by a constrained optimiser. Its purpose is to
    *demonstrate* the equivalence: fitted on simulated data it should recover
    the same parameters as ``arch``, and on real data it should forecast
    equivalently (paper Tables 2 and 3).

:class:`StructuralGARCHLSTM`
    The paper's headline contribution, Equation 7. The GARCH kernel is
    reinterpreted as the LSTM's **output gate**, and the cell state becomes a
    multiplicative controller on it:

    .. math::
        f_t &= \sigma_g(W_f\epsilon_{t-1} + U_f\sigma^2_{t-1} + b_f) \\
        i_t &= \sigma_g(W_i\epsilon_{t-1} + U_i\sigma^2_{t-1} + b_i) \\
        o_t &= K_{\text{garch}}(\epsilon_{t-1}, \sigma^2_{t-1};\Theta) \\
        \tilde c_t &= \sigma_c(W_c\epsilon_{t-1} + U_c\sigma^2_{t-1} + b_c) \\
        c_t &= f_t\odot c_{t-1} + i_t\odot\tilde c_t \\
        \sigma^2_t &= o_t\odot\big(1 + w\tanh(c_t)\big)

    Read the last line carefully, because it is what makes the design work:
    at :math:`w = 0` the model collapses **exactly** to its GARCH kernel, and
    increasing :math:`w` lets the LSTM magnify (:math:`c_t>0`) or damp
    (:math:`c_t<0`) the econometric forecast. The neural part is a modulation of
    a valid variance, never a replacement for it, so the output is positive by
    construction as long as :math:`|w| < 1`.

    Note also that the input is :math:`\epsilon_{t-1}`, not :math:`\epsilon_t`
    as in a textbook LSTM. The paper is explicit about why: :math:`\epsilon_t`
    cannot be sampled before :math:`\sigma_t` is known.

Reference
---------
Zhao, P., Zhu, H., Ng, W. S. H., & Lee, D. L. (2024). From GARCH to Neural
Network for Volatility Forecast. *AAAI 2024*. https://arxiv.org/abs/2402.06642
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..core.base import VolatilityModel
from ..core.config import ModelConfig
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance
from ..distributions import Normal, StudentT, StudentTKernel
from ..distributions.base import Distribution
from ..mean.models import MeanModel, get_mean_model
from .kernels import GARCHKernel, get_kernel

__all__ = ["GARCHNN", "StructuralGARCHLSTM"]

_ZHAO = dict(
    paper_id="zhao2024_garch_nn",
    title="From GARCH to Neural Network for Volatility Forecast",
    authors=("Pengfei Zhao", "Haoren Zhu", "Wilfred Siu Hung Ng", "Dik Lun Lee"),
    year=2024,
    venue="AAAI 2024",
    paper_url="https://arxiv.org/abs/2402.06642",
    code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
    implementation_status=ImplementationStatus.PAPER_FAITHFUL,
)


class _ZhaoBase(VolatilityModel):
    """Shared plumbing: mean model, distribution, and the GARCH kernel."""

    def __init__(
        self,
        kernel: str | GARCHKernel = "gjr",
        config: ModelConfig | None = None,
        kernel_kwargs: dict | None = None,
    ):
        super().__init__(config)
        self._kernel = (
            kernel if isinstance(kernel, GARCHKernel)
            else get_kernel(kernel, numerical=self.config.numerical, **(kernel_kwargs or {}))
        )
        self._mean = get_mean_model(self.config.mean)

    @property
    def kernel(self) -> GARCHKernel:
        return self._kernel

    @property
    def cell(self):
        """The kernel stands in for a cell in the reporting machinery."""
        return self._kernel

    @property
    def mean_model(self) -> MeanModel:
        return self._mean

    def distribution(self) -> Distribution:
        name = self.config.distribution
        if name == "normal":
            return Normal()
        if name == "student-t":
            return StudentT(df=self._fixed_df)
        if name == "student-t-kernel":
            return StudentTKernel(df=self._fixed_df)
        raise ValueError(
            f"{type(self).__name__} supports 'normal', 'student-t' and "
            f"'student-t-kernel'; got {name!r}."
        )

    def supports_analytic_forecast(self) -> bool:
        return False

    def _initial_variance(self, eps: Tensor) -> Tensor:
        return eps.var(unbiased=True).clamp(min=self.config.numerical.variance_floor)

    # -- the recursion, shared by filter() and forecast() ------------------- #
    def _initial_memory(self, dtype: torch.dtype):
        """Recurrent memory beyond :math:`h_{t-1}`. The bare kernel has none."""
        return None

    def _step(self, eps_window: Tensor, h_prev: Tensor, memory):
        r"""One step :math:`h_{t-1} \to h_t`.

        ``eps_window`` holds the innovations up to :math:`\epsilon_{t-1}`, most
        recent last, and may carry leading batch dimensions (one per simulated
        path); ``h_prev`` and ``memory`` carry the same batch dimensions.
        """
        return self._kernel(eps_window, h_prev), memory

    def _recurse(self, eps: Tensor) -> tuple[Tensor, object]:
        """Filter the sample; return the variance path and the terminal memory."""
        lookback = self._kernel.lookback
        memory = self._initial_memory(eps.dtype)
        h = [self._initial_variance(eps)]
        for t in range(1, eps.numel()):
            h_t, memory = self._step(eps[max(0, t - lookback) : t], h[-1], memory)
            h.append(h_t)
        return torch.stack(h), memory

    def filter(self, r: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps, mu = self.mean_model.residuals(r)
        h, _ = self._recurse(eps)
        return h, eps, mu

    def _forecast_by_recursion(self, r, horizon, method, n_simulations, seed):
        r"""Exact one-step forecast, then Monte Carlo through the same recursion.

        :math:`h_{T+1}` is deterministic given the sample. Beyond it each path
        draws :math:`z\sim` the fitted innovation law, sets
        :math:`\epsilon = z\sqrt{h}`, slides it into the kernel's window and
        steps the recursion, all paths at once.
        """
        eps, _ = self.mean_model.residuals(r)
        h, memory = self._recurse(eps)
        window = eps[-self._kernel.lookback :]
        h_next, memory = self._step(window, h[-1], memory)
        if horizon == 1:
            return h_next.reshape(1), None, method

        generator = torch.Generator(device=h_next.device)
        generator.manual_seed(seed if seed is not None else torch.seed() % (2**31))
        u = torch.rand(n_simulations, horizon - 1, generator=generator, dtype=h_next.dtype)
        z = self.distribution().ppf(u.clamp(1e-9, 1 - 1e-9)).to(h_next.dtype)

        window = window.expand(n_simulations, -1).clone()
        current = h_next.expand(n_simulations).clone()
        if memory is not None:
            memory = memory.expand(n_simulations, *memory.shape).clone()
        paths = torch.empty(n_simulations, horizon, dtype=h_next.dtype)
        for k in range(horizon):
            paths[:, k] = current
            if k == horizon - 1:
                break
            shock = z[:, k] * torch.sqrt(current)
            window = torch.cat([window[:, 1:], shock[:, None]], dim=1)
            current, memory = self._step(window, current, memory)
        return paths.mean(dim=0), paths, method


class GARCHNN(_ZhaoBase):
    r"""The neural-network counterpart of a GARCH-family model.

    Structurally identical to the corresponding classical model; what differs is
    that it is estimated by backpropagation through the recursion rather than by
    a constrained optimiser such as SLSQP. Zhao et al. use this to *validate*
    the equivalence: Table 2 shows the NN counterpart recovering the same
    parameters as ``arch`` on simulated data, and Table 3 shows it forecasting
    equivalently on real data.

    That equivalence has a practical payoff the paper points out: once a
    GARCH-family model has an NN counterpart, its parameters can be estimated by
    autodiff, and no bespoke estimation algorithm has to be written for each new
    family member.

    Parameters
    ----------
    kernel
        ``"garch"``, ``"gjr"`` or ``"figarch"``, or a
        :class:`~garchai.structural.kernels.GARCHKernel` instance.
    config
        ``distribution="normal"`` is the paper's N-loss; ``"student-t-kernel"``
        with ``df`` fixed at 3 or 5 is its T-loss, which is what the paper
        actually trains on (Table 4 selects ``nu = 5``).
    df
        Degrees of freedom for the Student-t losses. Held **fixed**, as in the
        paper; the T-loss kernel drops normalising constants that depend on
        ``nu``, so estimating ``nu`` under it would be inconsistent.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> model = GARCHNN(kernel="garch")
    >>> model.kernel.stylized_fact
    'volatility clustering'
    """

    provenance = Provenance(
        **_ZHAO,
        equations="Eq. 1 (GARCH), Eq. 3 (FI-GARCH expansion), Eqs. 5-6 (N-loss, T-loss)",
        deviations=(
            "The paper names the GJR leverage coefficient lambda; garchai calls "
            "it gamma throughout for consistency with the wider literature.",
        ),
    )

    def __init__(
        self,
        kernel: str | GARCHKernel = "garch",
        config: ModelConfig | None = None,
        df: float = 5.0,
        kernel_kwargs: dict | None = None,
    ):
        super().__init__(kernel, config, kernel_kwargs)
        self._fixed_df = df


class StructuralGARCHLSTM(_ZhaoBase):
    r"""Zhao et al.'s GARCH-LSTM (Equation 7).

    The GARCH kernel replaces the LSTM's output gate; the cell state becomes a
    bounded multiplicative modulation of it. See the module docstring for the
    equations and for why the design keeps the variance positive.

    Parameters
    ----------
    kernel
        Which stylized fact to infuse. The paper's best results use ``"gjr"``.
    hidden_size
        The paper uses a **scalar** cell state, which is the restriction Wei et
        al. (2025) later criticise. ``hidden_size=1`` reproduces the paper.
        Larger values are a documented generalisation: the modulation is then
        averaged across units before being applied, so ``w = 0`` still collapses
        exactly to the kernel.
    w
        Starting value of the coupling weight. Constrained to ``(-1, 1)`` by a
        ``tanh``, which guarantees ``1 + w*tanh(c) > 0`` and hence a positive
        variance for any cell state.

    Examples
    --------
    >>> import torch
    >>> torch.set_default_dtype(torch.float64)
    >>> model = StructuralGARCHLSTM(kernel="gjr", w=0.0)
    >>> # at w = 0 the model must reproduce its GARCH kernel exactly
    >>> r = torch.randn(200) * 0.9
    >>> h_lstm, _, _ = model.filter(r)
    >>> plain = GARCHNN(kernel=model.kernel)
    >>> h_garch, _, _ = plain.filter(r)
    >>> bool(torch.allclose(h_lstm, h_garch, atol=1e-12))
    True
    """

    provenance = Provenance(
        **_ZHAO,
        equations="Eq. 7 (GARCH-LSTM)",
        deviations=(
            "hidden_size > 1 averages the cell-state modulation across units; "
            "the paper's specification is scalar (hidden_size=1).",
            "The coupling weight w is constrained to (-1, 1) via tanh so that "
            "1 + w*tanh(c_t) > 0 always. The paper does not state a constraint, "
            "but without one the variance can go negative.",
        ),
    )

    def __init__(
        self,
        kernel: str | GARCHKernel = "gjr",
        config: ModelConfig | None = None,
        hidden_size: int = 1,
        w: float = 0.1,
        df: float = 5.0,
        kernel_kwargs: dict | None = None,
    ):
        super().__init__(kernel, config, kernel_kwargs)
        self._fixed_df = df
        self.hidden_size = hidden_size
        dt = torch.float64

        # Gates read (eps_{t-1}, sigma^2_{t-1}) exactly as Eq. 7 specifies.
        self.W_f = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.U_f = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.b_f = nn.Parameter(torch.ones(hidden_size, dtype=dt))  # forget-gate bias 1
        self.W_i = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.U_i = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.b_i = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        self.W_c = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.U_c = nn.Parameter(torch.randn(hidden_size, dtype=dt) * 0.1)
        self.b_c = nn.Parameter(torch.zeros(hidden_size, dtype=dt))
        # bounded(), not a bare tanh: tanh saturates to exactly +/-1 in float64
        # once |raw| > 19, and w = +/-1 with tanh(c) = -/+1 would give a zero
        # variance. bounded() is nextafter-guarded, so |w| < 1 strictly.
        start = min(max((float(w) + 1.0) / 2.0, 1e-6), 1.0 - 1e-6)
        self.raw_w = nn.Parameter(torch.logit(torch.tensor(start, dtype=dt)))

    @property
    def coupling(self) -> Tensor:
        r"""The weight :math:`w`, constrained to ``(-1, 1)``.

        At ``w = 0`` the LSTM is switched off and the model is its GARCH kernel;
        as ``|w|`` grows the neural modulation matters more. Reading this after
        fitting tells you how much the network actually contributed.
        """
        from ..constraints.transforms import bounded

        return bounded(self.raw_w, -1.0, 1.0)

    def _initial_memory(self, dtype: torch.dtype) -> Tensor:
        return torch.zeros(self.hidden_size, dtype=dtype)   # c_0 = 0

    def _step(self, eps_window: Tensor, h_prev: Tensor, c: Tensor):
        # trailing unit axis so batched paths broadcast against the gate weights
        eps_prev = eps_window[..., -1:]
        h_in = h_prev[..., None]
        f = torch.sigmoid(self.W_f * eps_prev + self.U_f * h_in + self.b_f)
        i = torch.sigmoid(self.W_i * eps_prev + self.U_i * h_in + self.b_i)
        c_tilde = torch.tanh(self.W_c * eps_prev + self.U_c * h_in + self.b_c)
        c = f * c + i * c_tilde

        o = self._kernel(eps_window, h_prev)                # the GARCH output gate

        modulation = 1.0 + self.coupling * torch.tanh(c).mean(dim=-1)
        h = (o * modulation).clamp(min=self.config.numerical.variance_floor)
        return h, c
