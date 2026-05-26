#!/usr/bin/env python3
"""Create Python-native StaMPS parameter files.

Examples
--------
Convert an existing MATLAB parameter set:

    python create_parms_json.py --work-dir path/to/project --from-existing

Create a fresh PS/ISCE parameter file for a new project:

    python create_parms_json.py --work-dir path/to/project --processor isce
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from getparm import (
    StampsConfig,
    _CONDITIONAL_DEFAULTS,
    _STATIC_DEFAULTS,
    _value_to_json,
)

logger = logging.getLogger("stamps")


def _read_float(path: Path) -> Any:
    if not path.is_file():
        return "nan"
    try:
        arr = np.loadtxt(str(path))
        if np.ndim(arr) == 0:
            return float(arr)
        return arr.tolist()
    except Exception:
        return "nan"


def _read_processor(work_dir: Path, fallback: str) -> str:
    path = work_dir / "processor.txt"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback


def build_default_parms(
    work_dir: Path,
    processor: str = "isce",
    small_baseline: str = "n",
) -> dict[str, Any]:
    parms = dict(_STATIC_DEFAULTS)
    is_sb = small_baseline.strip().lower() == "y"
    for key, (sb_val, non_sb_val) in _CONDITIONAL_DEFAULTS.items():
        parms[key] = sb_val if is_sb else non_sb_val
    if is_sb:
        parms["sb_scla_drop_index"] = []

    parms["small_baseline_flag"] = "y" if is_sb else "n"
    parms["insar_processor"] = _read_processor(work_dir, processor)
    parms["lambda"] = _read_float(work_dir / "lambda.1.in")
    parms["heading"] = _read_float(work_dir / "heading.1.in")
    return parms


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_value_to_json(data), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logger.info("Wrote %s", path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Create parms.json/localparms.json for Python StaMPS.")
    parser.add_argument("--work-dir", required=True, help="Project directory")
    parser.add_argument(
        "--from-existing",
        action="store_true",
        help="Load existing parms.mat/parms.json with StampsConfig and write effective JSON.",
    )
    parser.add_argument("--processor", default="isce", choices=["isce", "doris", "gamma"])
    parser.add_argument("--small-baseline", default="n", choices=["y", "n"])
    parser.add_argument("--output", default="parms.json", help="Output filename/path, default parms.json in work-dir")
    parser.add_argument(
        "--local-output",
        default=None,
        help="Optional local overrides filename/path, e.g. localparms.json",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = work_dir / output

    if args.from_existing:
        StampsConfig.reset()
        cfg = StampsConfig(work_dir=work_dir)
        cfg.load()
        cfg.write_json(output, include_local=False)
        if args.local_output:
            local_output = Path(args.local_output)
            if not local_output.is_absolute():
                local_output = work_dir / local_output
            write_json(local_output, dict(cfg._local_parms))
        return 0

    parms = build_default_parms(
        work_dir=work_dir,
        processor=args.processor,
        small_baseline=args.small_baseline,
    )
    write_json(output, parms)
    if args.local_output:
        local_output = Path(args.local_output)
        if not local_output.is_absolute():
            local_output = work_dir / local_output
        write_json(local_output, {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
