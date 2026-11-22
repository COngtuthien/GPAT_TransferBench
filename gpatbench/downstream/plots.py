"""Plot-input validation (spec §20, §26). No renderer: no locked project environment has matplotlib.

§20: every plot is generated from CSV/Parquet outputs; no value is typed into a plotting script.
§26: plot scripts reject input tables that contain non-measured placeholders, except in template mode.
§19.1: the placeholder for an unmeasured cell is the em dash "—". "N/A" is a legitimate value (§25 M9 "N/A used
correctly"; §27 "metric shown N/A") and is not a placeholder. A missing cell (None or empty string) is treated as
unmeasured.
"""
from __future__ import annotations

from typing import Sequence

PLACEHOLDER = "—"


class PlaceholderInPlotInput(ValueError):
    pass


def validate_plot_table(rows: Sequence[dict], value_columns: Sequence[str], *, template_mode: bool) -> dict:
    """Refuse unmeasured cells in `value_columns` unless template_mode is True. Returns placeholder counts."""
    if not isinstance(template_mode, bool):
        raise TypeError("template_mode must be stated explicitly as a bool")
    found = []
    for i, r in enumerate(rows):
        for c in value_columns:
            if c not in r:
                raise PlaceholderInPlotInput(f"row {i}: missing column {c!r}")
            v = r[c]
            if v is None or (isinstance(v, str) and v.strip() in ("", PLACEHOLDER)):
                found.append((i, c))
    if found and not template_mode:
        raise PlaceholderInPlotInput(f"{len(found)} unmeasured cells, first {found[:5]}; only template mode may plot them")
    return {"rows": len(rows), "placeholder_cells": len(found), "template_mode": template_mode}
