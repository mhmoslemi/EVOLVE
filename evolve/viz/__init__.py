"""Headless, schema-aware plots generated from committed EVOLVE run artifacts."""

from .allocation import plot_allocation
from .archive import plot_archive
from .audits import plot_audits
from .provenance import plot_provenance
from .record import plot_record
from .roles import plot_roles
from .run import generate_plots

__all__ = [
    "generate_plots",
    "plot_allocation",
    "plot_archive",
    "plot_audits",
    "plot_provenance",
    "plot_record",
    "plot_roles",
]
