#!/usr/bin/env python3
"""Synthetic end-to-end check for prep_isce.py.

This script creates a tiny ISCE-like dataset, runs the Python preprocessor,
then verifies that ISCEPSLoader can consume the generated PATCH_1 files.
It is intentionally a standalone regression helper rather than a pytest test
because the repository does not yet have a formal test harness.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STAMPS_PYTHON = REPO_ROOT / "stamps_python"


def _write_complex64(path: Path, arr: np.ndarray) -> None:
    pairs = np.empty(arr.shape + (2,), dtype="<f4")
    pairs[..., 0] = arr.real.astype(np.float32)
    pairs[..., 1] = arr.imag.astype(np.float32)
    pairs.tofile(path)


def create_synthetic_isce(root: Path, width: int, length: int) -> Dict[str, Any]:
    """Create a small deterministic ISCE-like input tree."""
    isce = root / "isce"
    (isce / "master_20200101" / "master").mkdir(parents=True)
    slave_dates = ["20200113", "20200125", "20200206"]
    for date in slave_dates:
        (isce / date).mkdir(parents=True)

    rng = np.random.default_rng(9876)
    slc_paths = [isce / "master_20200101" / "master" / "master.slc"]
    slc_paths.extend(isce / date / "slave.slc" for date in slave_dates)

    base_amp = rng.uniform(80, 140, size=(length, width)).astype(np.float32)
    for path in slc_paths:
        amp = base_amp * rng.normal(1.0, 0.08, size=(length, width)).astype(np.float32)
        amp[amp < 1] = 1
        phase = rng.uniform(-np.pi, np.pi, size=(length, width)).astype(np.float32)
        _write_complex64(path, amp * np.exp(1j * phase))

    lon = 100 + np.arange(length * width, dtype=np.float32).reshape(length, width) / 1000
    lat = 30 + np.arange(length * width, dtype=np.float32).reshape(length, width) / 2000
    dem = 500 + np.arange(length * width, dtype=np.float32).reshape(length, width)
    lon.tofile(isce / "lon.raw")
    lat.tofile(isce / "lat.raw")
    dem.tofile(isce / "dem.raw")

    ifg_paths = []
    for date in slave_dates:
        real = rng.normal(0, 1, size=(length, width)).astype(np.float32)
        imag = rng.normal(0, 1, size=(length, width)).astype(np.float32)
        path = isce / date / "isce_minrefdem.int"
        _write_complex64(path, real + 1j * imag)
        ifg_paths.append(path)

    (isce / "width.txt").write_text(f"{width}\n", encoding="ascii")
    (isce / "len.txt").write_text(f"{length}\n", encoding="ascii")
    write_loader_metadata(isce, width, length)
    ifg_list = isce / "pscphase.in"
    ifg_list.write_text(
        f"{width}\n" + "".join(f"{path}\n" for path in ifg_paths),
        encoding="utf-8",
    )

    return {"isce": isce, "ifg_list": ifg_list, "slave_dates": slave_dates}


def write_loader_metadata(prepared: Path, width: int, length: int) -> None:
    """Write the minimal root metadata files required by ISCEPSLoader."""
    files = {
        "day.1.in": "20200113\n20200125\n20200206\n",
        "master_day.1.in": "20200101\n",
        "bperp.1.in": "10\n20\n30\n",
        "heading.1.in": "-169.4626649344252\n",
        "lambda.1.in": "0.056\n",
        "width.txt": f"{width}\n",
        "len.txt": f"{length}\n",
    }
    for name, text in files.items():
        (prepared / name).write_text(text, encoding="ascii")
    parms_src = REPO_ROOT / "test_data" / "parms.mat"
    if parms_src.is_file():
        shutil.copy2(parms_src, prepared / "parms.mat")


def run_check(output_root: Path, keep: bool = False) -> Dict[str, Any]:
    """Run the full synthetic prep + loader check."""
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    width = 10
    length = 8
    fixture = create_synthetic_isce(output_root, width, length)
    isce = fixture["isce"]
    prepared = output_root / "prepared"

    cmd = [
        sys.executable,
        str(STAMPS_PYTHON / "prep_isce.py"),
        str(isce),
        "--output",
        str(prepared),
        "--width",
        str(width),
        "--length",
        str(length),
        "--da-thresh",
        "0.5",
        "--range-patches",
        "1",
        "--azimuth-patches",
        "1",
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
        "--ifg-list",
        str(fixture["ifg_list"]),
        "--ifg-list-has-width",
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
    from data_loader import ISCEPSLoader  # pylint: disable=import-error,import-outside-toplevel
    from getparm import StampsConfig  # pylint: disable=import-error,import-outside-toplevel
    from stamps_main import StampsRunner  # pylint: disable=import-error,import-outside-toplevel

    StampsConfig.reset()
    loader = ISCEPSLoader(prepared / "PATCH_1")
    loader.load()
    ps1_path = prepared / "PATCH_1" / "ps1.h5"
    with h5py.File(ps1_path, "r") as hf:
        ps1_shape = list(hf["ph"].shape)
        ps1_n_ps = int(np.asarray(hf["n_ps"]).ravel()[0])
    ps1_mtime_before = ps1_path.stat().st_mtime_ns
    StampsConfig.reset()
    runner = StampsRunner(project_dir=prepared)
    runner.run(start_step=1, end_step=1)
    ps1_mtime_after = ps1_path.stat().st_mtime_ns

    report: Dict[str, Any] = {
        "output_root": str(output_root),
        "prep_returncode": proc.returncode,
        "prep_stdout_tail": proc.stdout[-1000:],
        "prep_stderr_tail": proc.stderr[-1000:],
        "n_ps": int(loader.n_ps),
        "n_ifg": int(loader.n_ifg),
        "n_image": int(loader.n_image),
        "ph_shape": list(loader.ph.shape),
        "lonlat_shape": list(loader.lonlat.shape),
        "hgt_shape": list(loader.hgt.shape),
        "bperp_mat_shape": list(loader.bperp_mat.shape),
        "master_ix": int(loader.master_ix),
        "day_len": int(loader.day.size),
        "candidate_file_count": len((prepared / "PATCH_1" / "pscands.1.ij").read_text().splitlines()),
        "calconst_len": int(loader.calconst.size),
        "ps1_exists": ps1_path.is_file(),
        "ps1_ph_shape": ps1_shape,
        "ps1_n_ps": ps1_n_ps,
        "runner_step1_reused_ps1": ps1_mtime_before == ps1_mtime_after,
    }

    expected = {
        "prep_returncode": 0,
        "n_ps": 80,
        "n_ifg": 4,
        "n_image": 4,
        "ph_shape": [80, 4],
        "lonlat_shape": [80, 2],
        "hgt_shape": [80],
        "bperp_mat_shape": [80, 3],
        "master_ix": 0,
        "day_len": 4,
        "candidate_file_count": 80,
        "calconst_len": 3,
        "ps1_exists": True,
        "ps1_ph_shape": [80, 4],
        "ps1_n_ps": 80,
        "runner_step1_reused_ps1": True,
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
        # Keep report but remove bulky generated inputs/outputs.
        for child in output_root.iterdir():
            if child.name != "report.json":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic prep_isce regression check.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "validation_runs" / "prep_isce_synthetic_check"),
        help="Directory for generated fixture and report.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep generated fixture files.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_check(Path(args.output_root).resolve(), keep=args.keep)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
