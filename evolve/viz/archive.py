"""Plot archive coverage over epochs and per-cell testing depth."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries, load_latest_checkpoint


def plot_archive(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    summaries = load_epoch_summaries(run_dir)
    checkpoint = load_latest_checkpoint(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if summaries:
        epochs = [item["epoch"] for item in summaries]
        coverage = [item.get("archive_coverage", 0.0) for item in summaries]
        axes[0].plot(epochs, coverage, marker="o")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("archive coverage")
        axes[0].set_title("Archive coverage")
    else:
        empty_figure_message(axes[0], "no committed epochs yet")

    archive = (checkpoint or {}).get("archive", {})
    cells = archive.get("cells", [])
    if cells:
        rewards_by_state = {
            state["state_id"]: state.get("internal_reward")
            for state in archive.get("states", [])
        }
        quality = sorted(
            (
                float(rewards_by_state[cell["champion_state_id"]])
                for cell in cells
                if cell.get("champion_state_id") in rewards_by_state
                and rewards_by_state[cell["champion_state_id"]] is not None
            ),
            reverse=True,
        )
        if quality:
            axes[1].bar(range(len(quality)), quality, color="tab:green")
            axes[1].set_xlabel("occupied cell (ranked)")
            axes[1].set_ylabel("confirmed champion reward")
            axes[1].set_title(f"Archive quality ({len(quality)} occupied cells)")
        else:
            empty_figure_message(axes[1], "no confirmed cell champions yet")
    else:
        empty_figure_message(axes[1], "no archive cells yet")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_archive"]
