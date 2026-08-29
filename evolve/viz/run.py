"""Schema-aware EVOLVE plots from committed run artifacts only.

    python -m evolve.viz.run RUN_DIR --all
    python -m evolve.viz.run RUN_DIR --record --archive

Every plot reads only already-committed JSON (``stepNN.summary.json``,
``checkpoints/latest.json``); it works after a run and while one is active,
and it never reruns candidate code or imports a model/CUDA library.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .allocation import plot_allocation
from .archive import plot_archive
from .audits import plot_audits
from .provenance import plot_provenance
from .record import plot_record
from .roles import plot_roles

_PLOTS = {
    "record": plot_record,
    "archive": plot_archive,
    "provenance": plot_provenance,
    "allocation": plot_allocation,
    "audits": plot_audits,
    "roles": plot_roles,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="an EVOLVE run directory")
    parser.add_argument("--all", action="store_true", help="generate every plot")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: RUN_DIR/plots)")
    for name in _PLOTS:
        parser.add_argument(f"--{name}", action="store_true", help=f"generate the {name} plot")
    return parser


def generate_plots(run_dir: Path, *, names: Sequence[str], out_dir: Path) -> Sequence[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        plot_fn = _PLOTS[name]
        try:
            path = plot_fn(run_dir, out_dir / f"{name}.png")
        except Exception as exc:  # a plot failure never aborts the rest
            print(f"evolve.viz.run: {name} plot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if path is not None:
            written.append(path)
    return written


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"evolve.viz.run: not a directory: {run_dir}", file=sys.stderr)
        return 2
    requested = [name for name in _PLOTS if args.all or getattr(args, name)]
    if not requested:
        requested = list(_PLOTS)
    out_dir = args.out.resolve() if args.out is not None else run_dir / "plots"
    written = generate_plots(run_dir, names=requested, out_dir=out_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_plots", "main"]
