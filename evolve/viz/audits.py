"""Plot audit pair volume per epoch and causal memory effect state."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries, load_latest_checkpoint

_STATUS_COLORS = {"promoted": "tab:green", "rejected": "tab:red", "quarantined": "tab:gray"}


def plot_audits(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    summaries = load_epoch_summaries(run_dir)
    checkpoint = load_latest_checkpoint(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if summaries:
        epochs = [item["epoch"] for item in summaries]
        pairs = [item.get("audit_pairs", 0) for item in summaries]
        axes[0].plot(epochs, pairs, marker="o", color="tab:orange")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("audit pairs")
        axes[0].set_title("Audit pairs per epoch")
    else:
        empty_figure_message(axes[0], "no committed epochs yet")

    records = (checkpoint or {}).get("causal_memory", [])
    if records:
        ranked = sorted(records, key=lambda item: item.get("effect_mean", 0.0), reverse=True)
        effects = [item.get("effect_mean", 0.0) for item in ranked]
        colors = [_STATUS_COLORS.get(item.get("status"), "tab:gray") for item in ranked]
        axes[1].bar(range(len(effects)), effects, color=colors)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_xlabel("memory record (ranked)")
        axes[1].set_ylabel("effect mean")
        axes[1].set_title("Causal memory effects (green=promoted, red=rejected)")
    else:
        empty_figure_message(axes[1], "no causal memory records yet")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_audits"]
