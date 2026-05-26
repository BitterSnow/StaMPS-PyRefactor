#!/usr/bin/env python3
"""Synthetic regression check for ISCE2 small-baseline preparation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STAMPS_PYTHON = REPO_ROOT / "stamps_python"


def _write_complex64(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = np.empty(arr.shape + (2,), dtype="<f4")
    pairs[..., 0] = arr.real.astype(np.float32)
    pairs[..., 1] = arr.imag.astype(np.float32)
    pairs.tofile(path)


def create_synthetic_isce2_sb(root: Path, width: int, length: int) -> Dict[str, Any]:
    """Create a tiny ISCE2-like stack with small-baseline IFGs."""
    isce = root / "isce" / "merged"
    dates = [20200101, 20200113, 20200125, 20200206]
    pairs = [(20200101, 20200113), (20200113, 20200125), (20200125, 20200206)]
    bperp = {20200113: 10.0, 20200125: 20.0, 20200206: 30.0}
    rng = np.random.default_rng(20260525)

    base_amp = rng.uniform(80, 140, size=(length, width)).astype(np.float32)
    yy, xx = np.indices((length, width), dtype=np.float32)
    for i, date in enumerate(dates):
        amp = base_amp * (1.0 + i * 0.01)
        phase = 0.1 * i + yy * 0.02 + xx * 0.03
        _write_complex64(
            isce / "SLC" / str(date) / f"{date}.slc.full",
            amp * np.exp(1j * phase),
        )

    for date, value in bperp.items():
        bdir = root / "isce" / "baselines" / f"20200101_{date}"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / f"20200101_{date}.txt").write_text(
            f"Bperp (average): {value}\n",
            encoding="ascii",
        )

    ifg_paths = []
    for d1, d2 in pairs:
        real = rng.normal(0, 1, size=(length, width)).astype(np.float32)
        imag = rng.normal(0, 1, size=(length, width)).astype(np.float32)
        path = isce / "interferograms" / f"{d1}_{d2}" / f"filt_{d1}_{d2}.int"
        _write_complex64(path, real + 1j * imag)
        ifg_paths.append(path)

    lon = 100 + np.arange(length * width, dtype=np.float32).reshape(length, width) / 1000
    lat = 30 + np.arange(length * width, dtype=np.float32).reshape(length, width) / 2000
    dem = 500 + np.arange(length * width, dtype=np.float32).reshape(length, width)
    lon.tofile(isce / "lon.raw")
    lat.tofile(isce / "lat.raw")
    dem.tofile(isce / "dem.raw")

    return {"isce": isce, "pairs": pairs, "ifg_paths": ifg_paths}


def run_check(output_root: Path, keep: bool = False) -> Dict[str, Any]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    width = 10
    length = 8
    fixture = create_synthetic_isce2_sb(output_root, width, length)
    isce = fixture["isce"]
    prepared = output_root / "prepared"

    cmd = [
        sys.executable,
        str(STAMPS_PYTHON / "prep_isce.py"),
        str(isce),
        "--output",
        str(prepared),
        "--small-baseline",
        "--ifg-dir",
        str(isce / "interferograms"),
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
        "--write-ps1",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    sys.path.insert(0, str(STAMPS_PYTHON))
    import h5py  # pylint: disable=import-outside-toplevel
    from data_loader import ISCESBLoader  # pylint: disable=import-error,import-outside-toplevel
    from getparm import StampsConfig  # pylint: disable=import-error,import-outside-toplevel

    StampsConfig.reset()
    loader = ISCESBLoader(prepared / "PATCH_1")
    if proc.returncode == 0:
        loader.load()

    ps1_path = prepared / "PATCH_1" / "ps1.h5"
    h5_info: Dict[str, Any] = {}
    if ps1_path.is_file():
        with h5py.File(ps1_path, "r") as hf:
            h5_info = {
                "ps1_ph_shape": list(hf["ph"].shape),
                "has_ifgday": "ifgday" in hf,
                "has_ifgday_ix": "ifgday_ix" in hf,
                "bperp": [float(x) for x in np.asarray(hf["bperp"][:]).ravel()],
            }

    ifgday_text = (prepared / "ifgday.1.in").read_text(encoding="ascii").strip().splitlines()
    pscphase_lines = (prepared / "pscphase.in").read_text(encoding="utf-8").strip().splitlines()
    small_baselines_text = (
        prepared / "small_baselines.list"
    ).read_text(encoding="ascii").strip().splitlines()

    report: Dict[str, Any] = {
        "output_root": str(output_root),
        "prep_returncode": proc.returncode,
        "prep_stdout_tail": proc.stdout[-1000:],
        "prep_stderr_tail": proc.stderr[-1000:],
        "ifgday": ifgday_text,
        "small_baselines": small_baselines_text,
        "pscphase_line_count": len(pscphase_lines),
        "pscphase_width": int(pscphase_lines[0]) if pscphase_lines else None,
        "n_ps": int(loader.n_ps) if proc.returncode == 0 else None,
        "n_ifg": int(loader.n_ifg) if proc.returncode == 0 else None,
        "n_image": int(loader.n_image) if proc.returncode == 0 else None,
        "ph_shape": list(loader.ph.shape) if proc.returncode == 0 else None,
        "bperp": [float(x) for x in loader.bperp] if proc.returncode == 0 else None,
        **h5_info,
    }

    expected_ifgday = ["20200101 20200113", "20200113 20200125", "20200125 20200206"]
    expected = {
        "prep_returncode": 0,
        "ifgday": expected_ifgday,
        "small_baselines": expected_ifgday,
        "pscphase_line_count": 4,
        "pscphase_width": width,
        "n_ps": width * length,
        "n_ifg": 3,
        "n_image": 4,
        "ph_shape": [width * length, 3],
        "ps1_ph_shape": [width * length, 3],
        "has_ifgday": True,
        "has_ifgday_ix": True,
        "bperp": [10.0, 10.0, 10.0],
    }
    failures = {
        key: {"expected": value, "actual": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    report["passed"] = not failures
    report["failures"] = failures

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
    parser = argparse.ArgumentParser(description="Run synthetic ISCE2 SB prep regression check.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "validation_runs" / "prep_isce_sb_synthetic_check"),
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
