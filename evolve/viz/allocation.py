"""Plot scheduler allocation by role over time and the latest reservation mix."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries


def plot_allocation(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    summaries = load_epoch_summaries(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if summaries:
        roles = sorted({role for item in summaries for role in item.get("arms_by_role", {})})
        epochs = [item["epoch"] for item in summaries]
        for role in roles:
            counts = [item.get("arms_by_role", {}).get(role, 0) for item in summaries]
            axes[0].plot(epochs, counts, marker="o", label=role)
        if roles:
            axes[0].legend()
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("planned arms")
        axes[0].set_title("Allocation by role")

        latest_slots = summaries[-1].get("reservation_slots", {})
        if latest_slots:
            labels = list(latest_slots.keys())
            values = [latest_slots[key] for key in labels]
            axes[1].barh(labels, values, color="tab:blue")
            axes[1].set_title(f"Reservation slots (epoch {summaries[-1]['epoch']})")
        else:
            empty_figure_message(axes[1], "no reservation data yet")
    else:
        empty_figure_message(axes[0], "no committed epochs yet")
        empty_figure_message(axes[1], "no committed epochs yet")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_allocation"]
