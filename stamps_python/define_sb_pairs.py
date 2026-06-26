#!/usr/bin/env python3
"""Define a Small Baseline pair graph before patch preparation.

The output is intentionally small and StaMPS-like: ``ifgday.1.in`` plus
``small_baselines.list``.  prep_isce.py and build_sb_phase.py should consume
the same file so candidate statistics and phase columns stay aligned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from prep_isce import (  # pylint: disable=wrong-import-position
    SmallBaselineIFG,
    discover_secondary_slcs,
    discover_small_baseline_ifgs,
    generate_consecutive_sb_pairs,
    infer_width_length,
    read_small_baseline_pair_list,
)


def _pair_dates(ifgs: Sequence[SmallBaselineIFG]) -> List[int]:
    return sorted({date for ifg in ifgs for date in (ifg.date1, ifg.date2)})


def _slc_dates(slcs: Sequence[Path]) -> set[int]:
    dates: set[int] = set()
    for path in slcs:
        for part in (path.stem, path.name, path.parent.name):
            if len(part) >= 8:
                for i in range(0, len(part) - 7):
                    text = part[i : i + 8]
                    if text.isdigit() and text.startswith(("19", "20")):
                        dates.add(int(text))
                        break
    return dates


def _validate_slc_dates(ifgs: Sequence[SmallBaselineIFG], slcs: Sequence[Path]) -> None:
    available = _slc_dates(slcs)
    missing = [date for date in _pair_dates(ifgs) if date not in available]
    if missing:
        raise FileNotFoundError(
            "Missing co-registered SLCs for SB pair date(s): "
            + ", ".join(str(date) for date in missing)
        )


def resolve_pairs(args: argparse.Namespace, slcs: Sequence[Path]) -> List[SmallBaselineIFG]:
    """Resolve the requested pair graph."""
    source = args.pair_source
    if source == "list":
        if not args.pair_list:
            raise ValueError("--pair-list is required when --pair-source=list")
        return read_small_baseline_pair_list(Path(args.pair_list).resolve())
    if source == "consecutive":
        return generate_consecutive_sb_pairs(slcs, neighbor_count=args.neighbor_count)
    if source == "ifg":
        return discover_small_baseline_ifgs(
            Path(args.slc_root).resolve(),
            ifg_dir=args.ifg_dir,
            ifg_pattern=args.ifg_pattern,
        )

    try:
        return discover_small_baseline_ifgs(
            Path(args.slc_root).resolve(),
            ifg_dir=args.ifg_dir,
            ifg_pattern=args.ifg_pattern,
        )
    except FileNotFoundError:
        print(
            "[define_sb_pairs] no existing IFGs found; generating consecutive pair graph",
            flush=True,
        )
        return generate_consecutive_sb_pairs(slcs, neighbor_count=args.neighbor_count)


def write_pair_files(output_dir: Path, ifgs: Sequence[SmallBaselineIFG], width: int | None) -> None:
    """Write pair graph files consumed by SB preparation and phase building."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_text = "".join(f"{ifg.date1} {ifg.date2}\n" for ifg in ifgs)
    (output_dir / "ifgday.1.in").write_text(pair_text, encoding="ascii")
    (output_dir / "small_baselines.list").write_text(pair_text, encoding="ascii")

    phase_paths = [ifg.path for ifg in ifgs if ifg.path is not None]
    if width is not None or phase_paths:
        first_line = f"{width if width is not None else 0}\n"
        body = "".join(f"{path.resolve()}\n" for path in phase_paths)
        (output_dir / "pscphase.in").write_text(first_line + body, encoding="utf-8")


def define_pairs(args: argparse.Namespace) -> None:
    slc_root = Path(args.slc_root).resolve()
    output_dir = Path(args.output).resolve()
    slcs = discover_secondary_slcs(slc_root)
    if not slcs:
        raise FileNotFoundError(f"No co-registered SLCs found from {slc_root}")

    ifgs = resolve_pairs(args, slcs)
    if not ifgs:
        raise ValueError("No SB pairs were resolved")
    _validate_slc_dates(ifgs, slcs)

    width = args.width
    if width is None and args.write_pscphase:
        width, _length = infer_width_length(slc_root)
    write_pair_files(output_dir, ifgs, width if args.write_pscphase else None)

    dates = _pair_dates(ifgs)
    print(
        f"[define_sb_pairs] wrote {len(ifgs)} pairs across {len(dates)} dates -> {output_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Define the SB pair graph consumed by prep_isce.py and build_sb_phase.py."
    )
    parser.add_argument(
        "--slc-root",
        required=True,
        help="ISCE2 merged stack directory containing SLC/YYYYMMDD/*.slc.full",
    )
    parser.add_argument("--output", required=True, help="Directory to write ifgday.1.in")
    parser.add_argument(
        "--pair-source",
        choices=["auto", "consecutive", "ifg", "list"],
        default="consecutive",
        help="Pair graph source",
    )
    parser.add_argument(
        "-c",
        "--neighbor-count",
        type=int,
        default=3,
        help="Number of forward neighboring acquisitions for consecutive pairs",
    )
    parser.add_argument("--pair-list", default=None, help="Two-column YYYYMMDD pair list")
    parser.add_argument("--ifg-dir", default=None, help="Directory containing ISCE2 SB interferograms")
    parser.add_argument(
        "--ifg-pattern",
        default="filt_*.int,fine.int,isce_minrefdem.int",
        help="Comma-separated IFG glob pattern priority list",
    )
    parser.add_argument("--width", type=int, default=None, help="Raster width for optional pscphase.in")
    parser.add_argument(
        "--write-pscphase",
        action="store_true",
        help="Also write pscphase.in with width and any discovered IFG paths",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    define_pairs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
