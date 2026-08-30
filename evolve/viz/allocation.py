"""Plot committed allocation by role, cell, option, and harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_allocation_plans


def plot_allocation(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    plans = load_allocation_plans(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    flat_axes = tuple(axes.flat)

    if plans:
        latest = plans[-1]
        fields = (
            ("role", "Role allocation"),
            ("cell_id", "Cell allocation"),
            ("option_id", "Option allocation"),
            ("harness_id", "Harness allocation"),
        )
        for axis, (field, title) in zip(flat_axes, fields):
            counts = {}
            for planned in latest.get("planned_arms", ()):
                label = str(planned.get("arm", {}).get(field, "unknown"))
                counts[label] = counts.get(label, 0) + int(planned.get("replicas", 1))
            if not counts:
                empty_figure_message(axis, "no committed allocations")
                continue
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            labels = [item[0] for item in ranked]
            values = [item[1] for item in ranked]
            axis.barh(range(len(labels)), values, color="tab:blue")
            axis.set_yticks(range(len(labels)), labels=labels)
            axis.invert_yaxis()
            axis.set_xlabel("branch replicas")
            axis.set_title(f"{title} (epoch {latest['epoch']})")
    else:
        for axis in flat_axes:
            empty_figure_message(axis, "no committed allocation plans yet")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_allocation"]
