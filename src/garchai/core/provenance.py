"""Paper-to-code traceability.

Every model class in ``garchai`` carries a :class:`Provenance` describing which
paper it implements, which equations, whether official code exists, and every
point at which the implementation departs from the published specification.

This is not documentation decoration. Blueprint sec. 43 makes it a requirement:
a reader must be able to tell, from the object itself, whether a model is a
faithful reproduction or an inspired generalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CodeStatus(str, Enum):
    """Availability of source code for the original paper (blueprint sec. 14)."""

    OFFICIAL = "OFFICIAL"
    AUTHOR_LINKED = "AUTHOR_LINKED"
    PAPER_LINKED = "PAPER_LINKED"
    PACKAGE_SOURCE = "PACKAGE_SOURCE"
    COMMUNITY_REPLICATION = "COMMUNITY_REPLICATION"
    INDEPENDENT_RELATED = "INDEPENDENT_RELATED"
    NO_PUBLIC_CODE_IDENTIFIED = "NO_PUBLIC_CODE_IDENTIFIED"
    UNKNOWN = "UNKNOWN"


class ImplementationStatus(str, Enum):
    """How close this implementation is to the original (blueprint sec. 43)."""

    PAPER_FAITHFUL = "PAPER_FAITHFUL"
    """Follows the published equations exactly; deviations are cosmetic."""

    GENERALIZED = "GENERALIZED"
    """The paper's model is the default, but the class admits wider settings."""

    REPRODUCTION = "REPRODUCTION"
    """Rebuilt from the paper without official code to check against."""

    INSPIRED = "INSPIRED"
    """Motivated by the paper, but not a claim of reproduction."""

    EXPERIMENTAL = "EXPERIMENTAL"
    """Our own extension with no single source paper."""


@dataclass(frozen=True)
class Provenance:
    """Where a model comes from and how far it strays.

    Examples
    --------
    >>> p = Provenance(
    ...     paper_id="wei2025_garch_rnn",
    ...     title="Unified GARCH-Recurrent Neural Networks in Financial "
    ...           "Volatility Forecasting",
    ...     authors=("Jingyi Wei", "Steve Yang", "Zhenyu Cui"),
    ...     year=2025,
    ...     paper_url="https://arxiv.org/abs/2504.09380",
    ...     equations="Eqs. 15-21 (v2)",
    ...     code_status=CodeStatus.NO_PUBLIC_CODE_IDENTIFIED,
    ...     implementation_status=ImplementationStatus.PAPER_FAITHFUL,
    ... )
    >>> p.paper_id
    'wei2025_garch_rnn'
    """

    paper_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    paper_url: str | None = None
    doi: str | None = None
    equations: str | None = None
    """Which equations of the paper this class implements, e.g. ``"Eqs. 8a-8d"``."""

    code_status: CodeStatus = CodeStatus.UNKNOWN
    official_repo: str | None = None
    community_repos: tuple[str, ...] = ()
    implementation_status: ImplementationStatus = ImplementationStatus.REPRODUCTION
    deviations: tuple[str, ...] = ()
    """Every point where this code differs from the paper. Empty means none."""

    data_substitutions: tuple[str, ...] = ()
    """Datasets replaced because the original is unavailable. See docs/03."""

    notes: str | None = None

    def citation(self) -> str:
        """A one-line human-readable citation."""
        who = ", ".join(self.authors) if self.authors else "Unknown"
        bits = [f"{who} ({self.year})" if self.year else who, self.title]
        if self.venue:
            bits.append(self.venue)
        ref = self.doi or self.paper_url
        if ref:
            bits.append(ref)
        return ". ".join(bits)

    def summary(self) -> str:
        """Multi-line block suitable for printing under ``model.summary()``."""
        lines = [
            "Provenance",
            "----------",
            f"  Paper          : {self.citation()}",
            f"  Equations      : {self.equations or 'n/a'}",
            f"  Code status    : {self.code_status.value}",
            f"  Implementation : {self.implementation_status.value}",
        ]
        if self.official_repo:
            lines.append(f"  Official repo  : {self.official_repo}")
        for i, repo in enumerate(self.community_repos):
            label = "  Community      :" if i == 0 else "                  "
            lines.append(f"{label} {repo}")
        if self.deviations:
            lines.append("  Deviations from the paper:")
            lines.extend(f"    - {d}" for d in self.deviations)
        else:
            lines.append("  Deviations     : none")
        if self.data_substitutions:
            lines.append("  Data substitutions:")
            lines.extend(f"    - {d}" for d in self.data_substitutions)
        if self.notes:
            lines.append(f"  Notes          : {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True)
class DerivedProvenance(Provenance):
    """Provenance for a model that is our own extension of a published idea."""

    implementation_status: ImplementationStatus = ImplementationStatus.EXPERIMENTAL
    inspired_by: tuple[str, ...] = field(default=())
