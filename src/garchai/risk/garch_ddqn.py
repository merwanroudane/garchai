r"""Pokou, Kamdem & Benhmidene (2025): VaR via GARCH + Double Deep Q-Network.

The paper's move is to stop treating Value-at-Risk as a *regression* on a quantile
and treat it as a **classification** problem: an agent labels each day low-risk
(0) or high-risk (1), and the VaR threshold is then set per regime. The reason to
bother is class imbalance. High-risk days are rare by construction — that is what
makes them high-risk — and a supervised loss averaged over all days is dominated
by the calm majority, so it learns to say "calm" and is right 95% of the time
while being useless exactly when it matters.

Reinforcement learning lets the *reward* carry the asymmetry instead of a
reweighted loss (paper Eq. 12):

.. math::
    R_{t+1} = \begin{cases}
        +1     & \text{true positive — a detected crisis} \\
        -1     & \text{false positive — a false alarm} \\
        +\rho  & \text{true negative} \\
        -\rho  & \text{false negative — a missed crisis}
    \end{cases}

with :math:`\rho = n_{\text{minority}}/n_{\text{majority}} \in (0, 1]`. Because
:math:`\rho` is small, the *majority* class is the one that gets shrunk: calling a
calm day calm earns little, so the agent cannot coast. Note that a missed crisis
is penalised by :math:`-\rho`, not :math:`-1` — the paper's prose says a false
negative "incurs the most severe penalty", which its own equation does not do.
See ``deviations``; both conventions are available via ``severe_false_negative``.

Double DQN (Eq. 22) decouples action *selection* from action *evaluation*:

.. math::
    y_t = R_{t+1} + \gamma\,
          \hat{Q}\big(s_{t+1},\ \arg\max_a Q(s_{t+1}, a; \theta);\ \theta^-\big)

A single network does both jobs with the same weights, so any upward noise in a
Q-estimate is both chosen and believed — the classic overestimation bias. Using
the target network for the second job breaks that correlation.

Reference
---------
Pokou, F., Kamdem, J. S., & Benhmidene, A. (2025). Bridging Econometrics and AI:
VaR Estimation via Reinforcement Learning and GARCH Models. *Computational
Economics*. https://doi.org/10.1007/s10614-025-11191-w
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ..core.config import TrainConfig, set_seed
from ..core.provenance import CodeStatus, ImplementationStatus, Provenance

__all__ = ["GARCHDDQNVaR", "QNetwork", "ReplayBuffer", "Transition"]


@dataclass(frozen=True)
class Transition:
    """One step of experience."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Uniform experience replay.

    Its role here is not only decorrelation. Because minority-class days are
    rare, sampling uniformly *from the buffer* gives the agent repeated exposure
    to crisis episodes it would otherwise see once and forget.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 0):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Feed-forward Q-function :math:`Q(s, \\cdot\\,; \\theta)` over two actions."""

    def __init__(self, state_dim: int, hidden_sizes: tuple[int, ...] = (64, 32), n_actions: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = state_dim
        for size in hidden_sizes:
            layers += [nn.Linear(previous, size), nn.ReLU()]
            previous = size
        layers.append(nn.Linear(previous, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class DDQNHistory:
    """Training trace — enough to plot learning curves for a paper."""

    reward: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)


class GARCHDDQNVaR:
    r"""GARCH volatility + a DDQN risk-regime classifier, combined into VaR.

    Parameters
    ----------
    alpha
        VaR tail probability (0.01 and 0.05 are the usual regulatory choices).
    quantile_threshold
        A day is labelled high-risk if its return falls below this empirical
        quantile of the training returns. The paper uses the VaR level itself.
    severe_false_negative
        ``False`` (default) follows the paper's **equation**: a false negative
        earns :math:`-\rho`. ``True`` follows the paper's **prose**, making a
        missed crisis the harshest outcome at :math:`-1/\rho`. The two disagree;
        see ``provenance.deviations``.

    Examples
    --------
    >>> from garchai.risk import GARCHDDQNVaR
    >>> agent = GARCHDDQNVaR(alpha=0.05)
    >>> agent.alpha
    0.05
    """

    provenance = Provenance(
        paper_id="rl_var_ddqn_2025",
        title=(
            "Bridging Econometrics and AI: VaR Estimation via Reinforcement Learning "
            "and GARCH Models"
        ),
        authors=("Fredy Pokou", "Jules Sadefo Kamdem", "Abdeslam Benhmidene"),
        year=2025,
        paper_url="https://doi.org/10.1007/s10614-025-11191-w",
        equations="Eq. 12 (reward), Eq. 13-19 (MDP), Eq. 20-21 (DQN), Eq. 22 (DDQN)",
        code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
        implementation_status=ImplementationStatus.REPRODUCTION,
        deviations=(
            "The paper's prose states that a false negative 'incurs the most severe "
            "penalty', but its Eq. 12 assigns the false negative -rho, which with "
            "rho < 1 is *milder* than the false positive's -1. We implement Eq. 12 "
            "as written (the default) and expose severe_false_negative=True for the "
            "prose reading. This is a genuine internal inconsistency in the source, "
            "not an ambiguity we resolved silently.",
            "The regime-conditional quantile multipliers are estimated from the "
            "training set by empirical quantile of standardised residuals within "
            "each predicted regime; the paper does not state how it maps the "
            "classification to a threshold.",
            "State features (lagged returns, GARCH sigma, realised volatility) "
            "follow the paper's description; the exact feature list is not tabulated.",
        ),
        data_substitutions=(
            "The paper uses 16 years of EURO STOXX 50. Any index return series may "
            "be supplied; the bundled example uses a public equity index.",
        ),
    )

    def __init__(
        self,
        alpha: float = 0.05,
        window: int = 250,
        n_lags: int = 5,
        gamma: float = 0.95,
        hidden_sizes: tuple[int, ...] = (64, 32),
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        target_update_every: int = 50,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_anneal_fraction: float = 0.6,
        refit_every: int = 1,
        quantile_threshold: float | None = None,
        severe_false_negative: bool = False,
        seed: int = 0,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {alpha}.")
        self.alpha = alpha
        self.window = window
        self.n_lags = n_lags
        self.gamma = gamma
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        if not 0.0 < epsilon_anneal_fraction <= 1.0:
            raise ValueError("epsilon_anneal_fraction must lie in (0, 1].")
        self.epsilon_anneal_fraction = epsilon_anneal_fraction
        self.quantile_threshold = quantile_threshold if quantile_threshold is not None else alpha
        self.severe_false_negative = severe_false_negative
        self.seed = seed
        if refit_every < 1:
            raise ValueError("refit_every must be at least 1.")
        self.refit_every = refit_every
        self._sigma_cache = None

        self.buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self.online: QNetwork | None = None
        self.target: QNetwork | None = None
        self.rho = 1.0
        self.history = DDQNHistory()
        self._multipliers: dict[int, float] = {}
        self._mean = self._scale = None

    # -- environment -------------------------------------------------------- #
    def build_states(self, returns, sigma=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Assemble the state matrix, the risk labels, and the aligned returns.

        The state is lagged returns, the GARCH conditional volatility, and a
        short realised volatility — all observable at time :math:`t`, so nothing
        here can see the return it is asked to classify.
        """
        r = np.asarray(returns, dtype=float)
        if sigma is None:
            sigma = self._garch_sigma(r)
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != r.shape:
            raise ValueError("returns and sigma must align.")

        lags = np.column_stack([r[i : r.size - self.n_lags + i] for i in range(self.n_lags)])
        aligned_sigma = sigma[self.n_lags :]
        realised = np.array(
            [r[max(0, t - 20) : t].std() for t in range(self.n_lags, r.size)]
        )
        states = np.column_stack([lags, aligned_sigma, realised])
        target_returns = r[self.n_lags :]

        threshold = np.quantile(target_returns, self.quantile_threshold)
        labels = (target_returns < threshold).astype(int)
        return states, labels, target_returns

    def _garch_sigma(self, returns: np.ndarray) -> np.ndarray:
        r"""Rolling one-step GARCH(1,1) volatility — the econometric half.

        Cached on the input series. ``fit``, ``predict_regime``, ``var`` and
        ``evaluate`` are normally called on the *same* returns, and each one
        rebuilds the state matrix; without the cache the identical O(T) sequence
        of maximum-likelihood fits is paid four times over.

        ``refit_every = k`` re-estimates every k-th step and filters forward in
        between. It never uses future data, but the parameters lag by up to k
        observations, so it is an approximation to the exact protocol.
        """
        key = (returns.shape, float(returns[0]), float(returns[-1]), float(returns.sum()))
        if self._sigma_cache is not None and self._sigma_cache[0] == key:
            return self._sigma_cache[1]

        from arch import arch_model

        out = np.full(returns.size, np.nan)
        state = None
        for t in range(self.window, returns.size):
            if (t - self.window) % self.refit_every == 0 or state is None:
                try:
                    fitted = arch_model(
                        returns[t - self.window : t], mean="Constant", vol="GARCH", p=1, q=1
                    ).fit(disp="off", show_warning=False)
                    state = (fitted.params, float(fitted.conditional_volatility[-1] ** 2))
                except Exception:  # noqa: BLE001
                    out[t] = out[t - 1] if t > self.window else returns[:t].std()
                    continue
            parameters, variance = state
            residual = returns[t - 1] - parameters["mu"]
            variance = (
                parameters["omega"]
                + parameters["alpha[1]"] * residual**2
                + parameters["beta[1]"] * variance
            )
            state = (parameters, variance)
            out[t] = float(np.sqrt(variance))
        out[: self.window] = np.nanstd(returns[: self.window])
        self._sigma_cache = (key, out)
        return out

    def reward(self, action: int, label: int) -> float:
        """Paper Eq. 12, with the documented alternative for the prose reading."""
        if action == 1 and label == 1:
            return 1.0
        if action == 1 and label == 0:
            return -1.0
        if action == 0 and label == 0:
            return self.rho
        # false negative — a missed crisis
        return -1.0 / self.rho if self.severe_false_negative else -self.rho

    # -- training ----------------------------------------------------------- #
    def fit(
        self,
        returns,
        sigma=None,
        *,
        train: TrainConfig | None = None,
        n_episodes: int = 30,
    ) -> GARCHDDQNVaR:
        """Train the agent by sweeping the series once per episode, in order."""
        train = train or TrainConfig(lr=1e-3)
        set_seed(self.seed)

        states, labels, target_returns = self.build_states(returns, sigma)
        self._mean = states.mean(axis=0)
        self._scale = np.where(states.std(axis=0) > 1e-12, states.std(axis=0), 1.0)
        standardised = (states - self._mean) / self._scale

        minority = int(labels.sum())
        majority = int(labels.size - minority)
        if minority == 0 or majority == 0:
            raise ValueError(
                "Both risk classes must be present; adjust quantile_threshold."
            )
        self.rho = minority / majority

        self.online = QNetwork(standardised.shape[1], self.hidden_sizes)
        self.target = QNetwork(standardised.shape[1], self.hidden_sizes)
        self.target.load_state_dict(self.online.state_dict())
        optimiser = torch.optim.Adam(self.online.parameters(), lr=train.lr)

        # Anneal epsilon to its floor over a fixed fraction of the run, rather
        # than by a fixed per-episode rate. A rate tuned for one n_episodes
        # leaves the agent almost fully random for another: at 0.995 per episode
        # it is still 86% random after 30 episodes, so the reward curve reports
        # exploration noise rather than learning. DQN is off-policy, so the Q
        # function still trains -- which is exactly why the bug is easy to miss.
        anneal_episodes = max(int(self.epsilon_anneal_fraction * n_episodes), 1)
        decay = (
            (self.epsilon_end / self.epsilon_start) ** (1.0 / anneal_episodes)
            if self.epsilon_start > 0
            else 0.0
        )
        epsilon = self.epsilon_start
        step = 0
        for _ in range(n_episodes):
            episode_reward, losses, actions = 0.0, [], []
            for t in range(standardised.shape[0] - 1):
                state = standardised[t]
                action = self._act(state, epsilon)
                r = self.reward(action, int(labels[t]))
                episode_reward += r
                actions.append(action)
                self.buffer.push(
                    Transition(state, action, r, standardised[t + 1],
                               t == standardised.shape[0] - 2)
                )
                if len(self.buffer) >= self.batch_size:
                    losses.append(self._learn(optimiser))
                step += 1
                if step % self.target_update_every == 0:
                    self.target.load_state_dict(self.online.state_dict())

            predicted = np.array(actions)
            truth = labels[: predicted.size]
            positives = truth.sum()
            self.history.reward.append(episode_reward)
            self.history.loss.append(float(np.mean(losses)) if losses else float("nan"))
            self.history.epsilon.append(epsilon)
            self.history.accuracy.append(float((predicted == truth).mean()))
            self.history.recall.append(
                float(predicted[truth == 1].mean()) if positives else float("nan")
            )
            epsilon = max(self.epsilon_end, epsilon * decay)

        self._calibrate(standardised, target_returns, states[:, self.n_lags])
        return self

    def _act(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.get_default_dtype()))
        return int(q.argmax())

    def _learn(self, optimiser: torch.optim.Optimizer) -> float:
        """One DDQN update — paper Eq. 22."""
        batch = self.buffer.sample(self.batch_size)
        dtype = torch.get_default_dtype()
        states = torch.as_tensor(np.stack([b.state for b in batch]), dtype=dtype)
        next_states = torch.as_tensor(np.stack([b.next_state for b in batch]), dtype=dtype)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.long)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=dtype)
        done = torch.as_tensor([float(b.done) for b in batch], dtype=dtype)

        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Selection by the online net, evaluation by the target net — the
            # entire point of Double DQN.
            best = self.online(next_states).argmax(dim=1, keepdim=True)
            bootstrapped = self.target(next_states).gather(1, best).squeeze(1)
            y = rewards + self.gamma * (1.0 - done) * bootstrapped

        loss = nn.functional.mse_loss(q, y)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 5.0)
        optimiser.step()
        return float(loss.detach())

    # -- VaR ---------------------------------------------------------------- #
    def _calibrate(self, states: np.ndarray, returns: np.ndarray, sigma: np.ndarray) -> None:
        """Regime-conditional quantile multipliers from standardised residuals."""
        actions = self.predict_regime_from_states(states)
        safe_sigma = np.where(np.abs(sigma) > 1e-12, sigma, 1.0)
        standardised = returns / safe_sigma
        for regime in (0, 1):
            mask = actions == regime
            # Fall back to the pooled quantile when a regime is barely populated —
            # a 3-observation quantile is not an estimate.
            sample = standardised[mask] if mask.sum() >= 30 else standardised
            self._multipliers[regime] = float(np.quantile(sample, self.alpha))

    def predict_regime_from_states(self, standardised_states: np.ndarray) -> np.ndarray:
        if self.online is None:
            raise RuntimeError("Agent has not been fitted; call .fit(returns) first.")
        with torch.no_grad():
            q = self.online(
                torch.as_tensor(standardised_states, dtype=torch.get_default_dtype())
            )
        return q.argmax(dim=1).numpy()

    def predict_regime(self, returns, sigma=None) -> np.ndarray:
        """Risk labels for a series: 0 = low risk, 1 = high risk."""
        states, _, _ = self.build_states(returns, sigma)
        return self.predict_regime_from_states((states - self._mean) / self._scale)

    def var(self, returns, sigma=None) -> np.ndarray:
        r"""Regime-conditional VaR: :math:`\mathrm{VaR}_t = q_{\alpha}^{(a_t)}\,\sigma_t`.

        Returned as a **negative** number in return units, the convention used
        throughout ``garchai.risk``, so a violation is ``r_t < VaR_t``.
        """
        if self.online is None:
            raise RuntimeError("Agent has not been fitted; call .fit(returns) first.")
        states, _, _ = self.build_states(returns, sigma)
        actions = self.predict_regime_from_states((states - self._mean) / self._scale)
        sigma_column = states[:, self.n_lags]
        multipliers = np.array([self._multipliers[int(a)] for a in actions])
        return multipliers * sigma_column

    def evaluate(self, returns, sigma=None) -> dict[str, float]:
        """Classification quality *and* VaR backtests — the paper reports both."""
        from .backtests import christoffersen_conditional_coverage, kupiec_pof

        states, labels, target_returns = self.build_states(returns, sigma)
        actions = self.predict_regime_from_states((states - self._mean) / self._scale)
        var = self.var(returns, sigma)
        breaches = target_returns < var

        true_positive = int(((actions == 1) & (labels == 1)).sum())
        false_positive = int(((actions == 1) & (labels == 0)).sum())
        false_negative = int(((actions == 0) & (labels == 1)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)

        kupiec = kupiec_pof(target_returns, var, self.alpha)
        christoffersen = christoffersen_conditional_coverage(target_returns, var, self.alpha)
        return {
            "accuracy": float((actions == labels).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "violation_rate": float(breaches.mean()),
            "expected_rate": self.alpha,
            "kupiec_p": float(kupiec.p_value),
            "christoffersen_p": float(christoffersen.p_value),
            "mean_VaR": float(var.mean()),
        }
