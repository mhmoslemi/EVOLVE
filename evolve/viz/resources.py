"""Plot the committed global ledger and observed verifier resources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_latest_checkpoint


def plot_resources(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    checkpoint = load_latest_checkpoint(Path(run_dir))
    ledger = (checkpoint or {}).get("budget_ledger", {})
    evidence = (checkpoint or {}).get("archive", {}).get("evidence", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    limits = ledger.get("limits", {})
    consumed = {resource: 0.0 for resource in limits}
    for transaction in ledger.get("transactions", ()):
        resource = transaction.get("resource")
        if resource not in consumed:
            continue
        sign = 1.0 if transaction.get("kind") == "debit" else -1.0
        consumed[resource] += sign * float(transaction.get("amount", 0.0))
    if limits:
        labels = sorted(limits)
        fractions = [
            (float(consumed.get(label, 0.0)) / float(limits[label]))
            if float(limits[label]) > 0.0 else 0.0
            for label in labels
        ]
        axes[0].barh(labels, fractions, color="tab:blue")
        axes[0].axvline(1.0, linestyle="--", color="black", linewidth=0.8)
        axes[0].set_xlabel("fraction consumed")
        axes[0].set_title("Global resource ledger")
    else:
        empty_figure_message(axes[0], "no committed budget ledger")
    observed = {}
    for packet in evidence:
        for resource, amount in packet.get("resources", {}).items():
            observed[resource] = observed.get(resource, 0.0) + float(amount)
    if observed:
        labels = sorted(observed)
        axes[1].barh(labels, [observed[label] for label in labels], color="tab:green")
        axes[1].set_xlabel("observed total")
        axes[1].set_title("Verifier-observed resources")
    else:
        empty_figure_message(axes[1], "no observed resource evidence")
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


__all__ = ["plot_resources"]
