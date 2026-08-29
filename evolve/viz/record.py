"""Plot the confirmed record against epoch and cumulative verifier calls."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries


def plot_record(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    summaries = load_epoch_summaries(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if not summaries:
        empty_figure_message(axes[0], "no committed epochs yet")
        empty_figure_message(axes[1], "no committed epochs yet")
    else:
        epochs = [item["epoch"] for item in summaries]
        rewards = [item.get("confirmed_reward") for item in summaries]
        plotted_rewards = [value if value is not None else float("nan") for value in rewards]

        axes[0].plot(epochs, plotted_rewards, marker="o")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("confirmed internal reward")
        axes[0].set_title("Record vs epoch")

        cumulative_calls = []
        running = 0.0
        for item in summaries:
            running += float(item.get("costs", {}).get("verifier_calls", 0.0))
            cumulative_calls.append(running)
        axes[1].plot(cumulative_calls, plotted_rewards, marker="o", color="tab:orange")
        axes[1].set_xlabel("cumulative verifier calls")
        axes[1].set_ylabel("confirmed internal reward")
        axes[1].set_title("Record vs verifier calls")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_record"]
