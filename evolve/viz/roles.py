"""Plot each role's learning progress from the committed role checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_latest_checkpoint


def plot_roles(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    checkpoint = load_latest_checkpoint(run_dir)
    states = (checkpoint or {}).get("roles", {}).get("states", [])
    fig, ax = plt.subplots(figsize=(6, 4))

    if not states:
        empty_figure_message(ax, "no role checkpoint yet")
    else:
        roles = [state["role"] for state in states]
        revisions = [state.get("adapter", {}).get("revision", 0) for state in states]
        ax.bar(roles, revisions, color="tab:cyan")
        ax.set_ylabel("adapter revision (training updates applied)")
        ax.set_title("Role policy learning progress")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_roles"]
