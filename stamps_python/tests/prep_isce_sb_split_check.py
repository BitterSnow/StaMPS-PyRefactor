#!/usr/bin/env python3
"""Regression check for split SB prep + SLC-native phase building."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import h5py

from prep_isce_sb_synthetic_check import (  # pylint: disable=import-error
    REPO_ROOT,
    STAMPS_PYTHON,
    create_synthetic_isce2_sb,
)


def run_check(output_root: Path, keep: bool = False) -> Dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    width = 10
    length = 8
    fixture = create_synthetic_isce2_sb(output_root, width, length)
    isce = fixture["isce"]
    prepared = output_root / "prepared"

    define_cmd = [
        sys.executable,
        str(STAMPS_PYTHON / "define_sb_pairs.py"),
        "--slc-root",
        str(isce),
        "--output",
        str(prepared),
        "--pair-source",
        "consecutive",
        "-c",
        "1",
        "--width",
        str(width),
        "--write-pscphase",
    ]
    define_proc = subprocess.run(
        define_cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    prep_cmd = [
        sys.executable,
        str(STAMPS_PYTHON / "prep_isce.py"),
        str(isce),
        "--output",
        str(prepared),
        "--prepare",
        "sb",
        "--sb-pair-source",
        "list",
        "--sb-pair-list",
        str(prepared / "ifgday.1.in"),
        "--width",
        str(width),
        "--length",
        str(length),
        "--da-thresh",
        "10",
        "--range-overlap",
        "0",
        "--azimuth-overlap",
        "0",
        "--run-calamp",
        "--run-select",
        "--run-extract",
        "--skip-phase",
        "--lon",
        str(isce / "lon.raw"),
        "--lat",
        str(isce / "lat.raw"),
        "--dem",
        str(isce / "dem.raw"),
        "--master-date",
        "20200101",
        "--heading",
        "-169.4626649344252",
        "--wavelength",
        "0.056",
    ]
    prep_proc = subprocess.run(
        prep_cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ph_path = prepared / "PATCH_1" / "pscands.1.ph"
    phase_absent_after_prep = not ph_path.exists()

    build_cmd = [
        sys.executable,
        str(STAMPS_PYTHON / "build_sb_phase.py"),
        "--prepared",
        str(prepared),
        "--slc-root",
        str(isce),
        "--write-step1",
    ]
    build_proc = subprocess.run(
        build_cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ps1_path = prepared / "PATCH_1" / "ps1.h5"
    h5_info: Dict[str, Any] = {}
    if ps1_path.is_file():
        with h5py.File(ps1_path, "r") as f:
            h5_info = {
                "ph_shape": list(f["ph"].shape),
                "ifgday_shape": list(f["ifgday"].shape),
                "bperp": [float(x) for x in f["bperp"][:]],
            }

    report: Dict[str, Any] = {
        "output_root": str(output_root),
        "define_returncode": define_proc.returncode,
        "prep_returncode": prep_proc.returncode,
        "build_returncode": build_proc.returncode,
        "phase_absent_after_prep": phase_absent_after_prep,
        "phase_exists_after_build": ph_path.is_file(),
        **h5_info,
    }
    expected = {
        "define_returncode": 0,
        "prep_returncode": 0,
        "build_returncode": 0,
        "phase_absent_after_prep": True,
        "phase_exists_after_build": True,
        "ph_shape": [80, 3],
        "ifgday_shape": [3, 2],
        "bperp": [10.0, 10.0, 10.0],
    }
    failures = {key: {"expected": value, "got": report.get(key)} for key, value in expected.items() if report.get(key) != value}
    report["passed"] = not failures
    report["failures"] = failures
    report["define_stdout_tail"] = define_proc.stdout[-1000:]
    report["define_stderr_tail"] = define_proc.stderr[-1000:]
    report["prep_stdout_tail"] = prep_proc.stdout[-1000:]
    report["prep_stderr_tail"] = prep_proc.stderr[-1000:]
    report["build_stdout_tail"] = build_proc.stdout[-1000:]
    report["build_stderr_tail"] = build_proc.stderr[-1000:]

    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not keep and report["passed"]:
        for child in output_root.iterdir():
            if child.name != "report.json":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run split SB prep/build regression check.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "validation_runs" / "prep_isce_sb_split_check"),
    )
    parser.add_argument("--keep", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_check(Path(args.output_root).resolve(), keep=args.keep)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
