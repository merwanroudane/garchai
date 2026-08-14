r"""Variational Mode Decomposition (Dragomiretskiy & Zosso, 2013).

GENSHIN's second stage. VMD splits a signal into :math:`K` modes, each
narrow-band around its own centre frequency, by solving

.. math::
    \min_{\{u_k\},\{\omega_k\}}
      \sum_k \Big\|\partial_t\Big[
        \Big(\delta(t) + \tfrac{j}{\pi t}\Big) * u_k(t)\Big]e^{-j\omega_k t}
      \Big\|_2^2
    \quad\text{s.t.}\quad \sum_k u_k = f .

The bracketed term is the analytic signal of :math:`u_k`, shifted to baseband;
penalising its derivative is what forces each mode to be *compact around its own
frequency*. Solved by ADMM, which alternates three closed-form updates in the
Fourier domain — a Wiener-style filter for each mode, a power-weighted centroid
for each centre frequency, and a dual ascent step.

Why GENSHIN wants this rather than a wavelet or EMD: the modes come out
ordered, band-limited and reproducible, and the number of them is chosen rather
than discovered. Yu et al. decompose each of 60 fitted GARCH volatility series
into 4 modes, giving the 240 components that PLS then compresses.

Reference
---------
Dragomiretskiy, K., & Zosso, D. (2014). Variational Mode Decomposition.
*IEEE Transactions on Signal Processing* 62(3), 531-544.
"""

from __future__ import annotations

import numpy as np

__all__ = ["variational_mode_decomposition", "vmd_reconstruct"]


def variational_mode_decomposition(
    signal,
    n_modes: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
    init: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Decompose a 1-D signal into ``n_modes`` narrow-band components.

    Parameters
    ----------
    signal
        The series to decompose.
    n_modes
        :math:`K`. GENSHIN uses 4.
    alpha
        Bandwidth constraint. Larger values give narrower modes; 2000 is the
        usual default for signals normalised to unit sampling.
    tau
        Dual ascent step. ``0`` enforces the reconstruction constraint only
        loosely, which is the right choice for noisy data — with ``tau > 0`` the
        modes are forced to sum to the signal *exactly*, noise included.
    tolerance
        Convergence threshold on the summed relative mode change.
    init
        ``"uniform"`` spreads the initial centre frequencies evenly,
        ``"zero"`` starts them all at zero, ``"random"`` draws them.

    Returns
    -------
    modes, centre_frequencies
        ``modes`` has shape ``(n_modes, len(signal))`` ordered by increasing
        centre frequency; ``centre_frequencies`` are in cycles per sample.

    Notes
    -----
    The signal is **mirror-extended** before transforming and the extension
    discarded afterwards. Without it the implicit periodicity of the DFT creates
    a discontinuity at the boundary, which leaks energy across every mode — the
    decomposition still "works" and is quietly wrong at both ends.

    Examples
    --------
    Two well-separated tones separate cleanly:

    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1000, endpoint=False)
    >>> x = np.cos(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
    >>> modes, freqs = variational_mode_decomposition(x, n_modes=2, alpha=2000)
    >>> modes.shape
    (2, 1000)
    >>> bool(np.allclose(sorted(freqs * 1000), [5, 40], atol=1.5))
    True
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    if n < 4:
        raise ValueError("VMD needs at least 4 samples.")
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1.")

    # Mirror extension: kills the DFT's boundary discontinuity.
    half = n // 2
    extended = np.concatenate([x[:half][::-1], x, x[-half:][::-1]])
    total = extended.size

    frequencies = np.arange(total) / total - 0.5
    spectrum = np.fft.fftshift(np.fft.fft(extended))
    # Analytic signal: discard the negative half-spectrum.
    spectrum[: total // 2] = 0.0

    if init == "uniform":
        omega = np.linspace(0.5 / n_modes, 0.5, n_modes, endpoint=False)
    elif init == "zero":
        omega = np.zeros(n_modes)
    elif init == "random":
        rng = np.random.default_rng(0)
        omega = np.sort(np.exp(np.log(1.0 / total) + rng.random(n_modes) * (np.log(0.5) - np.log(1.0 / total))))
    else:
        raise ValueError(f"Unknown init {init!r}; use 'uniform', 'zero' or 'random'.")

    modes = np.zeros((n_modes, total), dtype=complex)
    dual = np.zeros(total, dtype=complex)

    for _ in range(max_iterations):
        previous = modes.copy()
        for k in range(n_modes):
            others = modes.sum(axis=0) - modes[k]
            # Wiener-style update: keep what the other modes have not claimed,
            # filtered by a bell centred on this mode's frequency.
            residual = spectrum - others + dual / 2.0
            modes[k] = residual / (1.0 + 2.0 * alpha * (frequencies - omega[k]) ** 2)

            power = np.abs(modes[k][total // 2 :]) ** 2
            denominator = power.sum()
            if denominator > 1e-12:
                omega[k] = (frequencies[total // 2 :] * power).sum() / denominator

        dual = dual + tau * (spectrum - modes.sum(axis=0))

        change = sum(
            np.sum(np.abs(modes[k] - previous[k]) ** 2)
            / max(np.sum(np.abs(previous[k]) ** 2), 1e-12)
            for k in range(n_modes)
        )
        if change < tolerance:
            break

    # Back to the time domain, restoring the discarded negative frequencies.
    output = np.zeros((n_modes, total))
    for k in range(n_modes):
        full = np.zeros(total, dtype=complex)
        full[total // 2 :] = modes[k][total // 2 :]
        full[1 : total // 2 + 1] = np.conj(modes[k][total - 1 : total // 2 - 1 : -1])
        full[0] = np.conj(full[-1])
        output[k] = np.real(np.fft.ifft(np.fft.ifftshift(full)))

    order = np.argsort(omega)
    return output[order][:, half : half + n], omega[order]


def vmd_reconstruct(modes: np.ndarray) -> np.ndarray:
    """Sum the modes back into a signal.

    With ``tau = 0`` the reconstruction is deliberately lossy — the residual is
    the noise VMD declined to assign to any band. Comparing this against the
    original is the honest way to see how much was discarded.
    """
    return np.asarray(modes).sum(axis=0)
