"""Implementation status: which papers are built, which are not.

Run it::

    python -m garchai.papers.status

The status is **derived**, never hand-maintained. It cross-references
``resources/papers.yaml`` against the ``Provenance`` objects actually declared
in the source tree, so a paper counts as built only when a model class claims it
by ``paper_id``. A hand-written checklist drifts from the code within a week;
this cannot.

Three states:

``BUILT``
    A model class declares this ``paper_id`` in its provenance.
``ADAPTER``
    Deliberately not rebuilt — ``hybridecon`` or another mature package already
    owns the model, and ``garchai`` reaches it through an adapter.
``TODO``
    Not yet implemented. ``blockers`` records anything preventing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

__all__ = ["PaperStatus", "collect", "summary_table", "render"]

_TIER = {
    "very_high": "S",
    "high": "A",
    "medium_high": "A",
    "medium": "B",
    "low": "C",
}


@dataclass
class PaperStatus:
    """Build status of a single paper."""

    paper_id: str
    title: str
    year: int | None
    tier: str
    state: str
    target_models: tuple[str, ...]
    built_models: tuple[str, ...]
    blockers: tuple[str, ...]
    covered_by: str | None

    @property
    def is_built(self) -> bool:
        return self.state == "BUILT"


def _source_root() -> Path:
    return Path(str(resources.files("garchai")))


def _declared_paper_ids(root: Path) -> set[str]:
    """Every ``paper_id`` claimed by a Provenance in the source tree."""
    ids: set[str] = set()
    for path in root.rglob("*.py"):
        ids |= set(re.findall(r'paper_id\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8")))
    return ids


def _declared_classes(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.py"):
        names |= set(re.findall(r"^class (\w+)", path.read_text(encoding="utf-8"), re.M))
    return names


def collect() -> list[PaperStatus]:
    """Build the status list by comparing the registry with the source tree."""
    import yaml

    root = _source_root()
    with open(root / "resources" / "papers.yaml", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)

    built_ids = _declared_paper_ids(root)
    classes = _declared_classes(root)

    out: list[PaperStatus] = []
    for paper in registry["papers"]:
        paper_id = paper["paper_id"]
        models = tuple(paper.get("models", ()))
        if paper_id in built_ids:
            state = "BUILT"
        elif paper.get("build_route") == "ADAPTER":
            state = "ADAPTER"
        else:
            state = "TODO"
        out.append(
            PaperStatus(
                paper_id=paper_id,
                title=paper.get("title", ""),
                year=paper.get("year"),
                tier=_TIER.get(paper.get("priority", ""), "?"),
                state=state,
                target_models=models,
                built_models=tuple(m for m in models if m in classes),
                blockers=tuple(paper.get("blockers", ()) or ()),
                covered_by=paper.get("covered_by"),
            )
        )
    return out


def summary_table():
    """Return the status as a :class:`pandas.DataFrame`."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "state": s.state,
                "tier": s.tier,
                "paper_id": s.paper_id,
                "year": s.year,
                "models": ", ".join(s.target_models),
                "built": ", ".join(s.built_models),
                "blockers": ", ".join(s.blockers),
            }
            for s in collect()
        ]
    )


def render() -> str:
    """A readable status report."""
    statuses = collect()
    order = {"BUILT": 0, "ADAPTER": 1, "TODO": 2}
    tier_order = {"S": 0, "A": 1, "B": 2, "C": 3, "?": 4}
    statuses.sort(key=lambda s: (order[s.state], tier_order[s.tier], s.paper_id))

    lines = ["garchai implementation status", "=" * 78, ""]
    current = None
    for s in statuses:
        if s.state != current:
            current = s.state
            counted = sum(1 for x in statuses if x.state == s.state)
            lines += [f"{s.state}  ({counted})", "-" * 78]
        detail = ""
        if s.blockers:
            detail = f"   [blocked: {', '.join(s.blockers)}]"
        elif s.covered_by:
            detail = f"   [{s.covered_by}]"
        lines.append(f"  Tier-{s.tier}  {s.paper_id:38s} {', '.join(s.target_models)}{detail}")
        if s.state == "BUILT" and set(s.built_models) != set(s.target_models):
            missing = set(s.target_models) - set(s.built_models)
            lines.append(f"           {'':38s} (registry lists but no class: {', '.join(sorted(missing))})")
    lines.append("")

    totals = {k: sum(1 for s in statuses if s.state == k) for k in ("BUILT", "ADAPTER", "TODO")}
    lines.append(
        f"TOTAL {len(statuses)} papers -> "
        f"BUILT {totals['BUILT']} | ADAPTER {totals['ADAPTER']} | TODO {totals['TODO']}"
    )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(render())


if __name__ == "__main__":  # pragma: no cover
    main()
