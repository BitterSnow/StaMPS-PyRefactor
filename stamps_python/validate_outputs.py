#!/usr/bin/env python3
"""
Compare StaMPS MATLAB .mat reference files with Python .h5 outputs.

The script is intentionally independent from the processing pipeline. It is
used as a validation aid while translating MATLAB steps to Python.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.io import loadmat


_DEFAULT_ALIASES: Dict[str, Sequence[str]] = {
    "bperp_mat": ("bperp_mat", "bperp"),
    "D_A": ("D_A", "D_A_mean", "D_A_std"),
    "ifg_std": ("ifg_std", "ifgstd"),
    "K_ps": ("K_ps", "K_ps_uw"),
    "C_ps": ("C_ps", "C_ps_uw"),
    "ph": ("ph", "ph_rc", "ph_patch", "ph_uw"),
    "psver": ("psver",),
}


@dataclass
class ArraySummary:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    finite: int
    nan: int
    inf: int
    mean_abs: Optional[float]
    std_abs: Optional[float]
    min_abs: Optional[float]
    max_abs: Optional[float]


@dataclass
class CompareResult:
    key: str
    mat_key: str
    h5_key: str
    mat: ArraySummary
    h5: ArraySummary
    comparable_shape: Tuple[int, ...]
    mean_abs_diff: Optional[float]
    max_abs_diff: Optional[float]
    rms_abs_diff: Optional[float]
    mean_phase_diff: Optional[float]
    max_phase_diff: Optional[float]
    note: str


def _is_matlab_private(name: str) -> bool:
    return name.startswith("__") or name in {"#refs#"}


def _decode_matlab_chars(arr: np.ndarray) -> Any:
    if arr.dtype.kind in ("U", "S"):
        if arr.ndim == 2 and arr.shape[0] == 1:
            return "".join(str(x) for x in arr.ravel()).strip()
        if arr.ndim <= 1:
            return "".join(str(x) for x in arr.ravel()).strip()
    return arr


def _normalize_array(value: Any) -> np.ndarray:
    value = _decode_matlab_chars(value) if isinstance(value, np.ndarray) else value
    arr = np.asarray(value)
    if arr.dtype.names and {"real", "imag"}.issubset(arr.dtype.names):
        arr = arr["real"] + 1j * arr["imag"]
    if arr.dtype.kind in ("S", "U", "O"):
        return arr
    arr = np.squeeze(arr)
    return np.asarray(arr)


def _read_h5_items(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    def visit(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            short_name = name.split("/")[-1]
            data = obj[()]
            out[name] = _normalize_array(data)
            out.setdefault(short_name, out[name])

    with h5py.File(path, "r") as hf:
        hf.visititems(visit)
    return out


def _read_mat_items(path: Path) -> Dict[str, np.ndarray]:
    try:
        raw = loadmat(path, squeeze_me=False, struct_as_record=False)
        return {
            k: _normalize_array(v)
            for k, v in raw.items()
            if not _is_matlab_private(k)
        }
    except NotImplementedError:
        return _read_h5_items(path)
    except ValueError as exc:
        msg = str(exc).lower()
        if "unknown mat file type" in msg or "hdf5" in msg or "7.3" in msg:
            return _read_h5_items(path)
        raise


def _as_numeric(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = _normalize_array(arr)
    if arr.dtype.kind not in ("b", "i", "u", "f", "c"):
        return None
    return arr


def _summary(name: str, arr: np.ndarray) -> ArraySummary:
    numeric = _as_numeric(arr)
    if numeric is None:
        return ArraySummary(
            name=name,
            shape=tuple(arr.shape),
            dtype=str(arr.dtype),
            finite=0,
            nan=0,
            inf=0,
            mean_abs=None,
            std_abs=None,
            min_abs=None,
            max_abs=None,
        )
    vals = np.abs(numeric.astype(np.complex128, copy=False)).ravel()
    finite_mask = np.isfinite(vals)
    finite_vals = vals[finite_mask]
    if finite_vals.size == 0:
        mean_abs = std_abs = min_abs = max_abs = None
    else:
        mean_abs = float(np.mean(finite_vals))
        std_abs = float(np.std(finite_vals))
        min_abs = float(np.min(finite_vals))
        max_abs = float(np.max(finite_vals))
    return ArraySummary(
        name=name,
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        finite=int(np.count_nonzero(finite_mask)),
        nan=int(np.count_nonzero(np.isnan(vals))),
        inf=int(np.count_nonzero(np.isinf(vals))),
        mean_abs=mean_abs,
        std_abs=std_abs,
        min_abs=min_abs,
        max_abs=max_abs,
    )


def _candidate_names(key: str) -> List[str]:
    names = [key]
    names.extend(_DEFAULT_ALIASES.get(key, ()))
    seen = set()
    result = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _find_key(items: Dict[str, np.ndarray], requested: str) -> Optional[str]:
    for name in _candidate_names(requested):
        if name in items:
            return name
    return None


def _shared_keys(mat_items: Dict[str, np.ndarray], h5_items: Dict[str, np.ndarray]) -> List[str]:
    mat_keys = {k for k in mat_items if not _is_matlab_private(k)}
    h5_keys = {k for k in h5_items if not _is_matlab_private(k)}
    return sorted(mat_keys & h5_keys)


def _sample_flat(arr: np.ndarray, limit: int) -> np.ndarray:
    flat = arr.ravel()
    if limit <= 0 or flat.size <= limit:
        return flat
    step = max(1, math.ceil(flat.size / limit))
    return flat[::step][:limit]


def _align_for_compare(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    a = _normalize_array(a)
    b = _normalize_array(b)
    note = ""
    if a.shape == b.shape:
        return a, b, note
    if a.T.shape == b.shape:
        return a.T, b, "mat transposed for shape match"
    if a.shape == b.T.shape:
        return a, b.T, "h5 transposed for shape match"
    if a.ndim == b.ndim == 2:
        at = a.T
        if at.shape[1:] == b.shape[1:]:
            n = min(at.shape[0], b.shape[0])
            return at[:n], b[:n], "mat transposed; compared common row prefix"
        bt = b.T
        if a.shape[1:] == bt.shape[1:]:
            n = min(a.shape[0], bt.shape[0])
            return a[:n], bt[:n], "h5 transposed; compared common row prefix"
        if a.shape[1] == b.shape[1]:
            n = min(a.shape[0], b.shape[0])
            return a[:n], b[:n], "row count mismatch; compared common row prefix"
        if a.shape[0] == b.shape[0]:
            n = min(a.shape[1], b.shape[1])
            return a[:, :n], b[:, :n], "column count mismatch; compared common column prefix"
    if a.ndim == b.ndim == 1:
        n = min(a.size, b.size)
        return a[:n], b[:n], "length mismatch; compared common prefix"
    n = min(a.size, b.size)
    return a.ravel()[:n], b.ravel()[:n], "shape mismatch; compared flattened common prefix"


def compare_key(
    key: str,
    mat_items: Dict[str, np.ndarray],
    h5_items: Dict[str, np.ndarray],
    sample_limit: int,
) -> Optional[CompareResult]:
    mat_key = _find_key(mat_items, key)
    h5_key = _find_key(h5_items, key)
    if mat_key is None or h5_key is None:
        return None

    mat_arr = mat_items[mat_key]
    h5_arr = h5_items[h5_key]
    mat_numeric = _as_numeric(mat_arr)
    h5_numeric = _as_numeric(h5_arr)
    mat_sum = _summary(mat_key, mat_arr)
    h5_sum = _summary(h5_key, h5_arr)
    if mat_numeric is None or h5_numeric is None:
        return CompareResult(
            key=key,
            mat_key=mat_key,
            h5_key=h5_key,
            mat=mat_sum,
            h5=h5_sum,
            comparable_shape=(),
            mean_abs_diff=None,
            max_abs_diff=None,
            rms_abs_diff=None,
            mean_phase_diff=None,
            max_phase_diff=None,
            note="non-numeric; summary only",
        )

    a, b, note = _align_for_compare(mat_numeric, h5_numeric)
    has_complex = np.iscomplexobj(a) or np.iscomplexobj(b)
    a_s = _sample_flat(a, sample_limit).astype(np.complex128, copy=False)
    b_s = _sample_flat(b, sample_limit).astype(np.complex128, copy=False)
    n = min(a_s.size, b_s.size)
    a_s = a_s[:n]
    b_s = b_s[:n]
    valid = np.isfinite(np.abs(a_s)) & np.isfinite(np.abs(b_s))
    if not np.any(valid):
        mean_abs_diff = max_abs_diff = rms_abs_diff = None
        mean_phase_diff = max_phase_diff = None
    else:
        diff = np.abs(a_s[valid] - b_s[valid])
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = float(np.max(diff))
        rms_abs_diff = float(np.sqrt(np.mean(diff ** 2)))
        if has_complex:
            phase_diff = np.abs(np.angle(a_s[valid] * np.conj(b_s[valid])))
            mean_phase_diff = float(np.mean(phase_diff))
            max_phase_diff = float(np.max(phase_diff))
        else:
            mean_phase_diff = max_phase_diff = None
    return CompareResult(
        key=key,
        mat_key=mat_key,
        h5_key=h5_key,
        mat=mat_sum,
        h5=h5_sum,
        comparable_shape=tuple(a.shape),
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=max_abs_diff,
        rms_abs_diff=rms_abs_diff,
        mean_phase_diff=mean_phase_diff,
        max_phase_diff=max_phase_diff,
        note=note,
    )


def _to_plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return {k: _to_plain(v) for k, v in value.__dict__.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def print_results(results: Sequence[CompareResult]) -> None:
    headers = [
        "key",
        "mat_shape",
        "h5_shape",
        "mean_diff",
        "max_diff",
        "rms_diff",
        "mean_phase",
        "note",
    ]
    print("\t".join(headers))
    for r in results:
        row = [
            r.key,
            str(r.mat.shape),
            str(r.h5.shape),
            _format_float(r.mean_abs_diff),
            _format_float(r.max_abs_diff),
            _format_float(r.rms_abs_diff),
            _format_float(r.mean_phase_diff),
            r.note,
        ]
        print("\t".join(row))


def _parse_key_map(items: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise ValueError("--map entries must use label=mat_key:h5_key")
        label, rest = item.split("=", 1)
        mat_key, h5_key = rest.split(":", 1)
        out[label] = (mat_key, h5_key)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare StaMPS MATLAB .mat reference files with Python .h5 outputs."
    )
    parser.add_argument("mat", type=Path, help="MATLAB .mat reference file")
    parser.add_argument("h5", type=Path, help="Python .h5 output file")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Variable names to compare. Defaults to shared keys.",
    )
    parser.add_argument(
        "--map",
        nargs="*",
        default=(),
        help="Explicit mappings like label=mat_key:h5_key.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200000,
        help="Maximum sampled elements per variable for diff statistics.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    mat_items = _read_mat_items(args.mat)
    h5_items = _read_h5_items(args.h5)
    mappings = _parse_key_map(args.map)

    results: List[CompareResult] = []
    for label, (mat_key, h5_key) in mappings.items():
        if mat_key not in mat_items or h5_key not in h5_items:
            continue
        one_mat = {label: mat_items[mat_key]}
        one_h5 = {label: h5_items[h5_key]}
        result = compare_key(label, one_mat, one_h5, args.sample)
        if result is not None:
            result.mat_key = mat_key
            result.h5_key = h5_key
            results.append(result)

    keys = args.keys if args.keys is not None else _shared_keys(mat_items, h5_items)
    for key in keys:
        result = compare_key(key, mat_items, h5_items, args.sample)
        if result is not None:
            results.append(result)

    print_results(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [_to_plain(r) for r in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
