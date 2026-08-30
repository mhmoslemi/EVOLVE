"""Plot provenance lineage branching from the committed archive snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_latest_checkpoint


def plot_provenance(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    checkpoint = load_latest_checkpoint(run_dir)
    edges = (checkpoint or {}).get("provenance", [])
    fig, ax = plt.subplots(figsize=(6, 4))

    if not edges:
        empty_figure_message(ax, "no provenance edges yet")
    else:
        record_state_id = (checkpoint or {}).get("record", {}).get("state_id")
        parent_by_child = {
            edge["child_state_id"]: edge["parent_state_id"] for edge in edges
        }
        record_lineage = set()
        cursor = record_state_id
        while cursor is not None and cursor not in record_lineage:
            record_lineage.add(cursor)
            cursor = parent_by_child.get(cursor)
        children_by_parent = {}
        for edge in edges:
            children_by_parent.setdefault(edge["parent_state_id"], []).append(edge["child_state_id"])
        ranked = sorted(
            children_by_parent.items(), key=lambda item: (-len(item[1]), item[0])
        )
        branching = [len(children) for _, children in ranked]
        colors = [
            "tab:orange" if parent in record_lineage else "tab:purple"
            for parent, _ in ranked
        ]
        ax.bar(range(len(branching)), branching, color=colors)
        ax.set_xlabel("parent state (ranked by descendant count)")
        ax.set_ylabel("descendant count")
        ax.set_title(
            f"Provenance branching ({len(edges)} edges; orange=record lineage)"
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_provenance"]
