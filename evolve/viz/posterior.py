"""Plot committed posterior admission and record-improvement calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ._common import empty_figure_message, load_latest_checkpoint


def _points(entries, field):
    points = []
    for entry in entries:
        beta = entry.get(field, {})
        prior_alpha = float(beta.get("prior_alpha", 1.0))
        prior_beta = float(beta.get("prior_beta", 1.0))
        successes = float(beta.get("successes", 0.0))
        failures = float(beta.get("failures", 0.0))
        support = successes + failures
        if support <= 0.0:
            continue
        alpha = prior_alpha + successes
        beta_value = prior_beta + failures
        points.append((alpha / (alpha + beta_value), successes / support, support))
    return points


def plot_posterior(run_dir: Union[str, Path], output_path: Union[str, Path]) -> Optional[Path]:
    checkpoint = load_latest_checkpoint(Path(run_dir))
    entries = (
        (checkpoint or {})
        .get("posterior", {})
        .get("levels", {})
        .get("arm", [])
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, field, title in (
        (axes[0], "reliability", "Infrastructure reliability"),
        (axes[1], "admission", "Admission calibration"),
        (axes[2], "improvement_given_admission", "Record-gain calibration"),
    ):
        points = _points(entries, field)
        if not points:
            empty_figure_message(axis, "no supported posterior observations yet")
            continue
        predicted, observed, support = zip(*points)
        axis.scatter(predicted, observed, s=[12.0 + 4.0 * value for value in support], alpha=0.7)
        axis.plot((0.0, 1.0), (0.0, 1.0), linestyle="--", color="black", linewidth=0.8)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("posterior probability")
        axis.set_ylabel("empirical rate")
        axis.set_title(title)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


__all__ = ["plot_posterior"]
