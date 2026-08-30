"""Plot the confirmed record against epochs, calls, tokens, and wall time."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_epoch_summaries


def plot_record(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    summaries = load_epoch_summaries(run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    flat_axes = tuple(axes.flat)
    if not summaries:
        for axis in flat_axes:
            empty_figure_message(axis, "no committed epochs yet")
    else:
        epochs = [item["epoch"] for item in summaries]
        rewards = [item.get("confirmed_reward") for item in summaries]
        plotted_rewards = [value if value is not None else float("nan") for value in rewards]

        cumulative = {"verifier_calls": [], "tokens": [], "wall_time_s": []}
        running = {key: 0.0 for key in cumulative}
        for item in summaries:
            costs = item.get("costs", {})
            exact_calls = item.get("budget_consumed", {}).get("verifier_calls")
            if exact_calls is None:
                running["verifier_calls"] += float(
                    costs.get("verifier_calls", 0.0)
                )
            else:
                # Barrier summaries expose the authoritative cumulative
                # ledger, including bootstrap, retries, confirmations and
                # refunds that are not branch-outcome costs.
                running["verifier_calls"] = float(exact_calls)
            running["tokens"] += float(costs.get("tokens", 0.0))
            running["wall_time_s"] += float(
                costs.get(
                    "verifier_wall_time_s",
                    costs.get("wall_time_s", costs.get("wall_time", 0.0)),
                )
            )
            for key in cumulative:
                cumulative[key].append(running[key])
        panels = (
            (epochs, "epoch", "tab:blue"),
            (cumulative["verifier_calls"], "cumulative verifier calls", "tab:orange"),
            (cumulative["tokens"], "cumulative generated tokens", "tab:green"),
            (
                cumulative["wall_time_s"],
                "cumulative verifier wall time (s)",
                "tab:red",
            ),
        )
        for axis, (x_values, label, color) in zip(flat_axes, panels):
            axis.plot(x_values, plotted_rewards, marker="o", color=color)
            axis.set_xlabel(label)
            axis.set_ylabel("confirmed internal reward")
            axis.set_title(f"Record vs {label}")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


__all__ = ["plot_record"]
