"""Headless, schema-aware plots generated from committed EVOLVE run artifacts."""

from pathlib import Path
from typing import Sequence

from .allocation import plot_allocation
from .archive import plot_archive
from .audits import plot_audits
from .provenance import plot_provenance
from .posterior import plot_posterior
from .record import plot_record
from .failures import plot_failures
from .resources import plot_resources
from .roles import plot_roles


def generate_plots(
    run_dir: Path, *, names: Sequence[str], out_dir: Path
) -> Sequence[Path]:
    """Generate plots without eagerly importing the ``-m`` entrypoint."""

    from .run import generate_plots as _generate_plots

    return _generate_plots(run_dir, names=names, out_dir=out_dir)


__all__ = [
    "generate_plots",
    "plot_allocation",
    "plot_archive",
    "plot_audits",
    "plot_provenance",
    "plot_posterior",
    "plot_record",
    "plot_failures",
    "plot_resources",
    "plot_roles",
]
