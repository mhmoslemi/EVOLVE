"""Plot committed scientific and infrastructure failure evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries, load_latest_checkpoint


def plot_failures(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    root = Path(run_dir)
    checkpoint = load_latest_checkpoint(root)
    summaries = load_epoch_summaries(root)
    evidence = ((checkpoint or {}).get("archive", {}).get("evidence", []))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = {}
    for packet in evidence:
        kind = str(packet.get("failure_kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    if counts:
        labels = sorted(counts)
        axes[0].bar(labels, [counts[label] for label in labels], color="tab:red")
        axes[0].tick_params(axis="x", rotation=35)
        axes[0].set_ylabel("evidence packets")
        axes[0].set_title("Verifier failure mix")
    else:
        empty_figure_message(axes[0], "no committed evidence yet")
    if summaries:
        epochs = [item["epoch"] for item in summaries]
        aborted = [item.get("infrastructure_aborted", item.get("latest_epoch", {}).get("infrastructure_aborted", 0)) for item in summaries]
        axes[1].plot(epochs, aborted, marker="o", color="tab:purple")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("infrastructure-aborted branches")
        axes[1].set_title("Infrastructure reliability")
    else:
        empty_figure_message(axes[1], "no committed epochs yet")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


__all__ = ["plot_failures"]
