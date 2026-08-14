r"""Publication-quality tables.

The build requirements ask for tables "suitable for research papers and strong
international academic journals". In practice that means a small number of
concrete conventions, applied consistently:

* point estimates with standard errors in parentheses beneath them,
* significance stars at the 10/5/1% levels, with the key in the note,
* the best value in each column marked in **bold**,
* an explicit note recording sample, estimator and anything substituted.

Output is available as LaTeX (``booktabs``), Markdown or plain text from the same
:class:`Table` object, so the same numbers go into a paper, a README and a
terminal without being retyped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Table", "estimation_table", "comparison_table", "significance_stars"]

STAR_LEVELS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))
STAR_NOTE = "*** p<0.01, ** p<0.05, * p<0.10."


def significance_stars(p_value: float) -> str:
    """Return the conventional star string for a p-value."""
    if not np.isfinite(p_value):
        return ""
    for threshold, stars in STAR_LEVELS:
        if p_value < threshold:
            return stars
    return ""


@dataclass
class Table:
    """A rendered table plus the metadata a journal will ask for."""

    frame: pd.DataFrame
    caption: str = ""
    note: str = ""
    label: str = ""
    bold_mask: pd.DataFrame | None = field(default=None, repr=False)

    def to_markdown(self) -> str:
        """Render as GitHub-flavoured Markdown."""
        body = self._decorated().to_markdown(index=True)
        parts = []
        if self.caption:
            parts.append(f"**{self.caption}**\n")
        parts.append(body)
        if self.note:
            parts.append(f"\n*Note:* {self.note}")
        return "\n".join(parts)

    def to_latex(self) -> str:
        """Render as a ``booktabs`` LaTeX table."""
        decorated = self._decorated()
        columns = "l" + "r" * decorated.shape[1]
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.caption}}}" if self.caption else "",
            f"\\label{{{self.label}}}" if self.label else "",
            f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule",
            " & ".join([decorated.index.name or ""] + list(map(str, decorated.columns)))
            + r" \\",
            r"\midrule",
        ]
        for name, row in decorated.iterrows():
            lines.append(" & ".join([str(name)] + [str(v) for v in row]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        if self.note:
            lines.append(
                r"\begin{tablenotes}\footnotesize" + f"\n\\item {self.note}\n"
                + r"\end{tablenotes}"
            )
        lines += [r"\end{table}"]
        return "\n".join(line for line in lines if line)

    def to_text(self) -> str:
        """Render as fixed-width plain text, for a terminal."""
        parts = [self.caption, "=" * max(len(self.caption), 40)] if self.caption else []
        parts.append(self._decorated().to_string())
        if self.note:
            parts.append(f"\nNote: {self.note}")
        return "\n".join(parts)

    def _decorated(self) -> pd.DataFrame:
        """Apply bold marking, if a mask was supplied."""
        if self.bold_mask is None:
            return self.frame
        out = self.frame.copy()
        for column in out.columns:
            if column in self.bold_mask.columns:
                flags = self.bold_mask[column]
                out[column] = [
                    f"**{value}**" if bool(flag) else value
                    for value, flag in zip(out[column], flags)
                ]
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.to_text()


def estimation_table(
    results: dict,
    standard_errors: dict[str, dict[str, float]] | None = None,
    p_values: dict[str, dict[str, float]] | None = None,
    caption: str = "GARCH-family estimation results",
    decimals: int = 4,
) -> Table:
    r"""Format one or more fitted models as a parameter table.

    Parameters
    ----------
    results
        ``{model_name: FitResult}``.
    standard_errors, p_values
        Optional ``{model_name: {parameter: value}}``. Standard errors are
        printed in parentheses on the row beneath the estimate, and p-values are
        turned into stars, which is the layout every econometrics journal
        expects.

    Returns
    -------
    Table
        Rows are parameters and diagnostics; columns are models.
    """
    names = list(results)
    parameters: list[str] = []
    for result in results.values():
        for key in list(result.parameters) + list(result.mean_parameters) + list(
            result.distribution_parameters
        ):
            if key not in parameters:
                parameters.append(key)

    rows: dict[str, list[str]] = {}
    for parameter in parameters:
        estimates, errors = [], []
        for name in names:
            result = results[name]
            merged = {
                **result.parameters,
                **result.mean_parameters,
                **result.distribution_parameters,
            }
            if parameter in merged:
                star = ""
                if p_values and name in p_values and parameter in p_values[name]:
                    star = significance_stars(p_values[name][parameter])
                estimates.append(f"{merged[parameter]:.{decimals}f}{star}")
                if standard_errors and name in standard_errors:
                    se = standard_errors[name].get(parameter)
                    errors.append(f"({se:.{decimals}f})" if se is not None else "")
                else:
                    errors.append("")
            else:
                estimates.append("")
                errors.append("")
        rows[parameter] = estimates
        if any(errors):
            rows[f"  {parameter} (s.e.)"] = errors

    for label, getter in [
        ("Log-likelihood", lambda r: f"{r.loglikelihood:.2f}"),
        ("AIC", lambda r: f"{r.aic:.2f}"),
        ("BIC", lambda r: f"{r.bic:.2f}"),
        ("Persistence", lambda r: f"{r.persistence:.4f}" if r.persistence else ""),
        ("Observations", lambda r: f"{r.n_obs:d}"),
    ]:
        rows[label] = [getter(results[name]) for name in names]

    frame = pd.DataFrame(rows, index=names).T
    frame.index.name = "Parameter"
    return Table(
        frame=frame,
        caption=caption,
        note=(
            "Maximum likelihood estimates. Standard errors in parentheses where "
            f"available. {STAR_NOTE}"
        ),
    )


def comparison_table(
    metrics: dict[str, dict[str, float]],
    caption: str = "Out-of-sample forecast accuracy",
    lower_is_better: tuple[str, ...] = ("MSE", "RMSE", "MAE", "QLIKE", "SMAPE", "Pinball"),
    decimals: int = 4,
) -> Table:
    r"""Format a model-by-metric comparison, marking the best value in each column.

    Parameters
    ----------
    metrics
        ``{model_name: {metric_name: value}}``.
    lower_is_better
        Metrics where a smaller value wins. Anything not listed (``R2``, for
        instance) is treated as higher-is-better. Getting this wrong silently
        bolds the *worst* model, so the list is explicit rather than inferred.
    """
    frame = pd.DataFrame(metrics).T
    frame.index.name = "Model"

    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for column in frame.columns:
        values = frame[column].astype(float)
        if not values.notna().any():
            continue
        best = values.idxmin() if column in lower_is_better else values.idxmax()
        mask.loc[best, column] = True

    formatted = frame.map(lambda v: f"{v:.{decimals}f}" if np.isfinite(v) else "")
    direction = ", ".join(m for m in lower_is_better if m in frame.columns)
    return Table(
        frame=formatted,
        caption=caption,
        note=(
            f"Best value per column in bold. Lower is better for: {direction}. "
            "Higher is better for all other columns."
        ),
        bold_mask=mask,
    )
