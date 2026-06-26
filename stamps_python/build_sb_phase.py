#!/usr/bin/env python3
"""Build Small Baseline phase samples from co-registered SLC pairs.

This script is intentionally separate from prep_isce.py.  prep_isce.py prepares
the StaMPS patch/candidate/geometry layout; this script fills each
PATCH_*/pscands.1.ph from an SLC-native SB pair graph stored in ifgday.1.in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from prep_isce import (  # pylint: disable=wrong-import-position
    PatchBounds,
    discover_secondary_slcs,
    extract_phase_from_slc_pairs,
    read_patch_bounds,
    read_scalar_int,
    write_ps1_h5_outputs,
)


def read_ifgday_pairs(path: Path) -> List[Tuple[int, int]]:
    """Read ifgday.1.in as sorted date pairs."""
    pairs: List[Tuple[int, int]] = []
    for line_no, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no} must contain two YYYYMMDD dates")
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise ValueError(f"No SB pairs found in {path}")
    return pairs


def discover_patches(prepared_dir: Path) -> List[PatchBounds]:
    """Return prepared PATCH_* entries in patch.list order when available."""
    patch_names: List[str]
    patch_list = prepared_dir / "patch.list"
    if patch_list.is_file():
        patch_names = [
            line.strip()
            for line in patch_list.read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
    else:
        patch_names = sorted(
            path.name
            for path in prepared_dir.glob("PATCH_*")
            if path.is_dir() and (path / "patch.in").is_file()
        )
    if not patch_names:
        raise FileNotFoundError(f"No PATCH_* directories found under {prepared_dir}")

    patches: List[PatchBounds] = []
    for name in patch_names:
        patch_in = prepared_dir / name / "patch.in"
        if not patch_in.is_file():
            raise FileNotFoundError(f"Missing patch bounds: {patch_in}")
        patches.append(read_patch_bounds(patch_in))
    return patches


def build_phase(args: argparse.Namespace) -> None:
    prepared_dir = Path(args.prepared).resolve()
    slc_root = Path(args.slc_root).resolve()
    width = args.width if args.width is not None else read_scalar_int(prepared_dir / "width.txt")
    pairs = read_ifgday_pairs(prepared_dir / "ifgday.1.in")
    patches = discover_patches(prepared_dir)
    slcs = discover_secondary_slcs(slc_root)
    if not slcs:
        raise FileNotFoundError(f"No co-registered SLCs found from {slc_root}")

    print(
        f"[build_sb_phase] {len(pairs)} pairs from ifgday.1.in, "
        f"{len(slcs)} SLCs, {len(patches)} patches",
        flush=True,
    )
    if not args.step1_only:
        for index, patch in enumerate(patches, start=1):
            patch_dir = prepared_dir / patch.name
            ij_path = patch_dir / "pscands.1.ij"
            if not ij_path.is_file():
                raise FileNotFoundError(f"Missing candidate coordinates: {ij_path}")
            print(
                f"[build_sb_phase] phase {patch.name} ({index}/{len(patches)})",
                flush=True,
            )
            extract_phase_from_slc_pairs(
                slcs,
                pairs,
                ij_path,
                width,
                patch_dir / "pscands.1.ph",
                precision=args.precision,
                byteswap=args.byteswap,
            )

    if args.write_step1 or args.step1_only:
        write_ps1_h5_outputs(prepared_dir, patches, small_baseline=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PATCH_*/pscands.1.ph for SLC-native StaMPS SB preparation.",
        epilog=(
            "SB pair count is controlled during prep_isce.py candidate preparation "
            "with --sb-pair-source/-c. This script consumes prepared ifgday.1.in "
            "so phase columns stay consistent with selsbc candidate statistics."
        ),
    )
    parser.add_argument(
        "--prepared",
        required=True,
        help="Prepared SB directory containing ifgday.1.in and PATCH_* folders",
    )
    parser.add_argument(
        "--slc-root",
        required=True,
        help="ISCE2 merged stack directory containing SLC/YYYYMMDD/*.slc.full",
    )
    parser.add_argument("--width", type=int, default=None, help="Raster width; defaults to prepared width.txt")
    parser.add_argument("--precision", choices=["f", "s"], default="f")
    parser.add_argument("--byteswap", action="store_true")
    parser.add_argument(
        "--step1-only",
        action="store_true",
        help="Only rewrite SB Step-1 HDF5 sidecars from existing pscands files",
    )
    parser.add_argument(
        "--write-step1",
        "--write-sb1",
        dest="write_step1",
        action="store_true",
        help="Write SB Step-1 HDF5 products after phase extraction",
    )
    parser.add_argument(
        "--write-ps1",
        dest="write_step1",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    build_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
