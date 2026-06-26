#!/usr/bin/env python3
"""
Headless StaMPS result export.

This module extracts the non-plotting parts of MATLAB ``ps_plot('v-*')`` and
``ts_flaghelper.m``:

* corrected unwrapped phase -> mean LOS velocity (mm/yr)
* corrected unwrapped phase -> displacement time series (mm)
* vector export to GeoPackage by default, with Shapefile retained as optional

The default correction is ``v-dso``:

    ph_uw - ph_scla - ph_scn_slave, then ps_deramp(...)

For the current Python workflow, the preferred input is the merged project
directory after Steps 1-8, containing ``ps2.h5``, ``phuw2.h5``, ``scla2.h5``
and optionally ``scn2.h5``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import h5py
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig  # noqa: E402

logger = logging.getLogger("stamps")


def _load_h5(path: Path, keys: Sequence[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as hf:
        for key in keys:
            if key in hf:
                out[key] = np.asarray(hf[key][:])
    return out


def _get_psver(project_dir: Path, default: int = 2) -> int:
    path = project_dir / "psver.h5"
    if not path.is_file():
        return default
    with h5py.File(str(path), "r") as hf:
        if "psver" in hf:
            return int(np.asarray(hf["psver"]).ravel()[0])
    return default


def _parse_drop_ifg(raw: Any, n_ifg: int) -> np.ndarray:
    if raw is None:
        return np.array([], dtype=np.int64)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "" or raw == "[]" or raw.lower() == "none":
            return np.array([], dtype=np.int64)
        try:
            import ast

            raw = ast.literal_eval(raw)
        except Exception:
            return np.array([], dtype=np.int64)
    try:
        arr = np.atleast_1d(np.asarray(raw).ravel())
        if arr.size == 0:
            return np.array([], dtype=np.int64)
        arr = np.round(arr.astype(np.float64)).astype(np.int64)
    except Exception:
        return np.array([], dtype=np.int64)
    arr = arr[(arr >= 1) & (arr <= n_ifg)]
    return (arr - 1).astype(np.int64)


def _yyyymmdd_to_ordinal(value: int) -> int:
    y = value // 10000
    m = (value % 10000) // 100
    d = value % 100
    return date(y, m, d).toordinal()


def _read_text_yyyymmdd(path: Path) -> list[int]:
    if not path.is_file():
        return []
    out: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip().split()
        if not text:
            continue
        token = text[0]
        if token.isdigit() and len(token) == 8:
            out.append(int(token))
    return out


def _infer_date_offset(project_dir: Path, raw_dates: np.ndarray, master_day: float) -> int:
    """Infer whether StaMPS serial dates are Python ordinals or MATLAB datenums.

    MATLAB datenums are exactly Python ordinals + 366 for modern dates.  Both
    serial systems fall in the same numeric range, so use the original text
    metadata when available instead of relying on plausibility.
    """
    text_dates = _read_text_yyyymmdd(project_dir / "day.1.in")
    text_dates.extend(_read_text_yyyymmdd(project_dir / "master_day.1.in"))
    if not text_dates:
        # Prefer MATLAB datenum for legacy StaMPS files; this also matches the
        # preprocessor sidecar text files used by current project directories.
        return 366

    raw_set = {int(round(float(x))) for x in np.asarray(raw_dates).ravel()}
    raw_set.add(int(round(float(master_day))))
    text_ord = {_yyyymmdd_to_ordinal(x) for x in text_dates}

    if raw_set & {x + 366 for x in text_ord}:
        return 366
    if raw_set & text_ord:
        return 0

    logger.warning(
        "Could not match HDF5 date serials to day.1.in/master_day.1.in; "
        "assuming MATLAB datenum offset"
    )
    return 366


def _serial_to_yyyymmdd(day_value: Union[int, float], offset: int) -> str:
    ordinal = int(round(float(day_value))) - offset
    return date.fromordinal(ordinal).strftime("%Y%m%d")


def _phase_to_mm(phase_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    return -phase_rad * (wavelength_m * 1000.0) / (4.0 * np.pi)


def _deramp_design_matrix(xy: np.ndarray, degree: float) -> np.ndarray:
    """Build MATLAB ``ps_deramp.m`` design matrix from ``ps.xy``."""
    if xy.ndim != 2:
        xy = xy.reshape(-1, 3)
    if xy.shape[1] < 3:
        raise ValueError("xy must have at least 3 columns: id, x, y")

    x = xy[:, 1].astype(np.float64) / 1000.0
    y = xy[:, 2].astype(np.float64) / 1000.0
    one = np.ones(x.shape[0], dtype=np.float64)

    if degree == 1:
        logger.info("Deramp model: z = ax + by + c")
        return np.column_stack([x, y, one])
    if degree == 1.5:
        logger.info("Deramp model: z = ax + by + cxy + d")
        return np.column_stack([x, y, x * y, one])
    if degree == 2:
        logger.info("Deramp model: z = ax^2 + by^2 + cxy + d")
        return np.column_stack([x**2, y**2, x * y, one])
    if degree == 3:
        logger.info("Deramp model: cubic")
        return np.column_stack(
            [x**3, y**3, x**2 * y, y**2 * x, x**2, y**2, x * y, one]
        )

    logger.warning("Unsupported deramp degree %s; falling back to degree 1", degree)
    return np.column_stack([x, y, one])


def _load_deramp_degree(project_dir: Path, default: float = 1.0) -> float:
    path = project_dir / "deramp_degree.mat"
    if not path.is_file():
        return default
    try:
        from getparm import load_mat

        data = load_mat(path)
        if "degree" in data:
            return float(np.asarray(data["degree"]).ravel()[0])
    except Exception as exc:
        logger.warning("Could not read %s: %s; using degree %.1f", path.name, exc, default)
    return default


def _ps_deramp_inplace(ph: np.ndarray, xy: np.ndarray, degree: float) -> np.ndarray:
    """Remove polynomial spatial ramp from each phase column.

    This mirrors MATLAB ``ps_deramp.m`` and returns the removed ramp.  ``ph`` is
    modified in place to avoid another full time-series copy.
    """
    A = _deramp_design_matrix(xy, degree)
    ramp = np.full(ph.shape, np.nan, dtype=np.float32)
    min_valid = A.shape[1] + 2

    for col in range(ph.shape[1]):
        y = ph[:, col]
        valid = np.isfinite(y)
        if int(valid.sum()) <= min_valid:
            logger.warning("IFG/date column %d not deramped: too few valid points", col + 1)
            continue
        coeff, _, _, _ = np.linalg.lstsq(A[valid], y[valid], rcond=None)
        col_ramp = A @ coeff
        ph[:, col] = y - col_ramp
        ramp[:, col] = col_ramp.astype(np.float32)
        if (col + 1) % 10 == 0 or (col + 1) == ph.shape[1]:
            logger.info("Deramped %d / %d columns", col + 1, ph.shape[1])

    return ramp


def _load_covariance(project_dir: Path, psver: int, unwrap_ix: np.ndarray) -> Optional[np.ndarray]:
    path = project_dir / f"ifgstd{psver}.h5"
    if not path.is_file():
        return None
    data = _load_h5(path, ["ifg_std"])
    if "ifg_std" not in data:
        return None
    ifg_std = np.asarray(data["ifg_std"], dtype=np.float64).ravel()
    if ifg_std.size <= int(unwrap_ix.max(initial=0)):
        return None
    var = (ifg_std[unwrap_ix] * np.pi / 180.0) ** 2
    var[~np.isfinite(var) | (var <= 0)] = 1.0
    return np.diag(var)


def _velocity_from_phase(
    ph: np.ndarray,
    day: np.ndarray,
    master_day: float,
    wavelength_m: float,
    covariance: Optional[np.ndarray],
) -> np.ndarray:
    """Compute mm/yr velocity using the MATLAB ps_plot linear model."""
    G = np.column_stack([np.ones(day.size), day.astype(np.float64) - master_day])

    if covariance is None or covariance.size == 0:
        solver = np.linalg.pinv(G)
    else:
        cov_inv = np.linalg.pinv(np.asarray(covariance, dtype=np.float64))
        solver = np.linalg.solve(G.T @ cov_inv @ G, G.T @ cov_inv)

    slope_weights = solver[1, :]
    slope = ph @ slope_weights
    return -slope * (365.25 / (4.0 * np.pi) * wavelength_m * 1000.0)


def _sb_pairs_to_single_master(
    ph_sb: np.ndarray,
    ifgday_ix: np.ndarray,
    n_image: int,
    master_ix_0: int,
) -> np.ndarray:
    """Invert SB pair phases to single-master acquisition phases.

    ``ifgday_ix`` is stored in MATLAB convention (1-based image indices).
    The master column is constrained to zero and omitted from the returned
    matrix.
    """
    ix0 = np.asarray(ifgday_ix, dtype=np.int64) - 1
    if ix0.ndim != 2 or ix0.shape[1] != 2:
        raise ValueError("ifgday_ix must be an n_ifg x 2 matrix")
    G = np.zeros((ix0.shape[0], n_image), dtype=np.float64)
    G[np.arange(ix0.shape[0]), ix0[:, 0]] = -1.0
    G[np.arange(ix0.shape[0]), ix0[:, 1]] = 1.0
    keep_cols = np.array([i for i in range(n_image) if i != master_ix_0], dtype=np.int64)
    G_sub = G[:, keep_cols]
    solver = np.linalg.pinv(G_sub)
    return ph_sb @ solver.T


def _select_reference_ps(lonlat: np.ndarray, xy: np.ndarray, cfg: StampsConfig) -> np.ndarray:
    try:
        from scla_estimation import _ps_setref

        ref_ps = _ps_setref(lonlat, xy, cfg)
        if ref_ps.size > 0:
            return ref_ps
    except Exception as exc:
        logger.warning("Reference area selection fell back to all PS: %s", exc)
    return np.arange(lonlat.shape[0], dtype=np.int64)


def _load_corrected_phase(
    project_dir: Path,
    psver: int,
    correction: str,
    apply_deramp: bool,
    deramp_degree: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    ps_path = project_dir / f"ps{psver}.h5"
    phuw_path = project_dir / f"phuw{psver}.h5"
    phuw_sb_path = project_dir / f"phuw_sb{psver}.h5"
    if not ps_path.is_file():
        raise FileNotFoundError(f"Need {ps_path.name} in {project_dir}")

    ps = _load_h5(
        ps_path,
        ["n_ps", "n_ifg", "n_image", "master_ix", "master_day", "day", "xy", "lonlat", "ifgday_ix"],
    )
    cfg = StampsConfig(work_dir=project_dir)
    small_baseline = str(cfg.getparm("small_baseline_flag") or "n").strip().lower() == "y"
    use_sb = small_baseline and not phuw_path.is_file() and phuw_sb_path.is_file()
    use_sm_from_sb = small_baseline and phuw_path.is_file()
    if use_sb:
        logger.info("SB export: inverting %s to single-master date phases", phuw_sb_path.name)
        ph_uw = _load_h5(phuw_sb_path, ["ph_uw"])["ph_uw"].astype(np.float64)
    else:
        if not phuw_path.is_file():
            raise FileNotFoundError(f"Need {phuw_path.name} or {phuw_sb_path.name} in {project_dir}")
        ph_uw = _load_h5(phuw_path, ["ph_uw"])["ph_uw"].astype(np.float64)

    lonlat = np.asarray(ps["lonlat"], dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(ps["xy"], dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(-1, 3)
    n_ifg = int(np.asarray(ps["n_ifg"]).ravel()[0])
    n_image = int(np.asarray(ps.get("n_image", [[n_ifg]])).ravel()[0])
    master_ix = int(np.asarray(ps["master_ix"]).ravel()[0]) - 1
    master_day = float(np.asarray(ps["master_day"]).ravel()[0])
    day_full = np.asarray(ps["day"], dtype=np.float64).ravel()

    correction = correction.lower()
    if correction not in {"v", "v-d", "v-do", "v-s", "v-so", "v-ds", "v-dso"}:
        raise ValueError("correction must be one of v, v-d, v-do, v-s, v-so, v-ds, v-dso")

    if "d" in correction:
        scla_path = project_dir / (f"scla_sb{psver}.h5" if use_sb else f"scla{psver}.h5")
        if not scla_path.is_file():
            raise FileNotFoundError(f"{correction} requires {scla_path.name}")
        scla = _load_h5(scla_path, ["ph_scla", "C_ps_uw"])
        ph_scla = np.asarray(scla["ph_scla"], dtype=np.float64)
        if ph_scla.shape != ph_uw.shape:
            raise ValueError(f"{scla_path.name}/ph_scla shape {ph_scla.shape} != ph_uw {ph_uw.shape}")
        ph_uw -= ph_scla
        if "C_ps_uw" in scla and correction == "v-d":
            # MATLAB v-d subtracts C before velocity; v-do/v-dso do not unless
            # ts_flag adds it.  The constant term does not affect velocity but
            # does shift time series, so keep this behaviour explicit.
            ph_uw -= np.asarray(scla["C_ps_uw"], dtype=np.float64).reshape(-1, 1)

    if "s" in correction:
        scn_path = project_dir / f"scn{psver}.h5"
        if not scn_path.is_file():
            raise FileNotFoundError(f"{correction} requires {scn_path.name}")
        scn = _load_h5(scn_path, ["ph_scn_slave"])
        ph_scn = np.asarray(scn["ph_scn_slave"], dtype=np.float64)
        if ph_scn.shape != ph_uw.shape:
            raise ValueError(f"{scn_path.name}/ph_scn_slave shape {ph_scn.shape} != ph_uw {ph_uw.shape}")
        ph_uw -= ph_scn

    if use_sb:
        ifgday_ix = np.asarray(ps.get("ifgday_ix"))
        if ifgday_ix.size == 0:
            raise FileNotFoundError("SB export requires ifgday_ix in ps2.h5")
        ph_uw = _sb_pairs_to_single_master(ph_uw, ifgday_ix, n_image, master_ix)
        image_ix = np.array([i for i in range(n_image) if i != master_ix], dtype=np.int64)
        day = day_full[image_ix]
        unwrap_ix = np.array([], dtype=np.int64)
    else:
        axis_count = ph_uw.shape[1]
        drop_ix = (
            np.array([], dtype=np.int64)
            if use_sm_from_sb
            else _parse_drop_ifg(StampsConfig(work_dir=project_dir).getparm("drop_ifg_index"), axis_count)
        )
        unwrap_ix = np.setdiff1d(np.arange(axis_count), drop_ix)
        unwrap_ix = np.setdiff1d(unwrap_ix, [master_ix])
        ph_uw = ph_uw[:, unwrap_ix]
        day = day_full[unwrap_ix]

    if apply_deramp:
        logger.info("Applying ps_deramp to corrected phase")
        ramp = _ps_deramp_inplace(ph_uw, xy, deramp_degree)
    else:
        ramp = np.empty((ph_uw.shape[0], 0), dtype=np.float32)

    ref_ps = _select_reference_ps(lonlat, xy, cfg)
    logger.info("Using %d reference PS", ref_ps.size)
    ph_uw -= np.nanmean(ph_uw[ref_ps, :], axis=0, keepdims=True)
    ph_uw = np.nan_to_num(ph_uw, nan=0.0)

    wavelength_m = float(np.asarray(cfg.getparm("lambda")).ravel()[0])
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        logger.warning("Invalid lambda in parms; using 0.056 m")
        wavelength_m = 0.056

    return ph_uw, lonlat, xy, day, master_day, unwrap_ix, ramp


def _build_full_time_series(
    ph_uw: np.ndarray,
    day: np.ndarray,
    master_day: float,
    wavelength_m: float,
    project_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    dates = np.array(sorted(set(int(round(d)) for d in day) | {int(round(master_day))}), dtype=np.float64)
    ph_mm = np.full((ph_uw.shape[0], dates.size), np.nan, dtype=np.float32)

    mm_without_master = _phase_to_mm(ph_uw, wavelength_m).astype(np.float32)
    master_int = int(round(master_day))
    master_col: Optional[int] = None
    for out_col, d in enumerate(dates):
        d_int = int(round(d))
        if d_int == master_int:
            master_col = out_col
            continue
        src = np.where(np.abs(day - d) < 0.5)[0]
        if src.size:
            ph_mm[:, out_col] = mm_without_master[:, src[0]]

    _fill_missing_time_series_values(ph_mm, dates, master_col=master_col)
    # Match the user's legacy script: sort dates, then make the first
    # acquisition the zero-displacement reference.
    ph_mm -= ph_mm[:, [0]]

    date_offset = _infer_date_offset(project_dir, dates, master_day)
    labels = [_serial_to_yyyymmdd(d, date_offset) for d in dates]
    output_dates = dates - date_offset
    output_master_day = float(master_day - date_offset)
    return ph_mm, output_dates, labels, output_master_day


def _fill_missing_time_series_values(
    ph_mm: np.ndarray,
    dates: np.ndarray,
    master_col: Optional[int] = None,
) -> None:
    """Fill missing export dates without creating one-epoch zero spikes.

    The reference/master acquisition is not an observed displacement column in
    PS mode, and can also be absent after SB inversion.  Older export code
    eventually converted these NaNs to zero, which creates a sharp artificial
    spike in otherwise smooth time series.  Fill missing columns from adjacent
    dates instead; for point-specific NaNs, use the same local interpolation
    where neighbouring values are finite.
    """
    n_col = ph_mm.shape[1]
    col_has_values = np.isfinite(ph_mm).any(axis=0)
    for col in range(n_col):
        missing = ~np.isfinite(ph_mm[:, col])
        if not bool(np.any(missing)):
            continue

        prev_cols = np.where(col_has_values[:col])[0]
        next_cols = np.where(col_has_values[col + 1 :])[0] + col + 1
        lo = int(prev_cols[-1]) if prev_cols.size else None
        hi = int(next_cols[0]) if next_cols.size else None

        if lo is not None and hi is not None:
            w = float((dates[col] - dates[lo]) / max(dates[hi] - dates[lo], 1.0))
            lo_vals = ph_mm[:, lo]
            hi_vals = ph_mm[:, hi]
            both = missing & np.isfinite(lo_vals) & np.isfinite(hi_vals)
            ph_mm[both, col] = ((1.0 - w) * lo_vals[both] + w * hi_vals[both]).astype(np.float32)
            only_lo = missing & np.isfinite(lo_vals) & ~np.isfinite(ph_mm[:, col])
            ph_mm[only_lo, col] = lo_vals[only_lo]
            only_hi = missing & np.isfinite(hi_vals) & ~np.isfinite(ph_mm[:, col])
            ph_mm[only_hi, col] = hi_vals[only_hi]
        elif lo is not None:
            lo_vals = ph_mm[:, lo]
            fill = missing & np.isfinite(lo_vals)
            ph_mm[fill, col] = lo_vals[fill]
        elif hi is not None:
            hi_vals = ph_mm[:, hi]
            fill = missing & np.isfinite(hi_vals)
            ph_mm[fill, col] = hi_vals[fill]

        remaining = ~np.isfinite(ph_mm[:, col])
        if bool(np.any(remaining)):
            label = "master/reference " if master_col == col else ""
            logger.warning(
                "Filled %d unresolved %sdate values in column %d with 0",
                int(np.sum(remaining)),
                label,
                col + 1,
            )
            ph_mm[remaining, col] = 0.0
        col_has_values[col] = bool(np.isfinite(ph_mm[:, col]).any())


def _write_plot_h5(
    output_dir: Path,
    velocity: np.ndarray,
    ph_mm: np.ndarray,
    lonlat: np.ndarray,
    dates: np.ndarray,
    date_labels: Sequence[str],
    master_day: float,
    correction: str,
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(output_dir / f"ps_plot_{correction}.h5"), "w") as hf:
        hf.create_dataset("ph_disp", data=velocity.astype(np.float32))
        hf.create_dataset("lonlat", data=lonlat.astype(np.float64))
        hf.attrs["correction"] = correction
    with h5py.File(str(output_dir / f"ps_plot_ts_{correction}.h5"), "w") as hf:
        hf.create_dataset("ph_mm", data=ph_mm.astype(np.float32))
        hf.create_dataset("lonlat", data=lonlat.astype(np.float64))
        hf.create_dataset("day", data=dates.astype(np.float64))
        hf.create_dataset("date", data=np.asarray(date_labels, dtype=object), dtype=str_dtype)
        hf.create_dataset("master_day", data=np.array(master_day, dtype=np.float64))
        hf.attrs["correction"] = correction
        hf.attrs["date_serial"] = "python_ordinal"


def _delete_vector(path: Path, driver_name: str) -> None:
    try:
        from osgeo import ogr

        driver = ogr.GetDriverByName(driver_name)
        if driver is not None and path.exists():
            driver.DeleteDataSource(str(path))
    except Exception:
        if path.exists():
            path.unlink()


def _create_field(layer: Any, name: str) -> None:
    from osgeo import ogr

    field = ogr.FieldDefn(name, ogr.OFTReal)
    field.SetWidth(18)
    field.SetPrecision(6)
    layer.CreateField(field)


def _write_vector_ogr(
    path: Path,
    driver_name: str,
    layer_name: str,
    lonlat: np.ndarray,
    velocity: np.ndarray,
    ph_mm: np.ndarray,
    date_labels: Sequence[str],
    chunk_size: int = 25000,
) -> None:
    from osgeo import ogr, osr

    _delete_vector(path, driver_name)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"OGR driver not available: {driver_name}")

    ds = driver.CreateDataSource(str(path))
    if ds is None:
        raise RuntimeError(f"Could not create {path}")

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    layer = ds.CreateLayer(layer_name, srs=sr, geom_type=ogr.wkbPoint)
    _create_field(layer, "vel")
    for label in date_labels:
        _create_field(layer, "D" + label)

    feature_def = layer.GetLayerDefn()
    n_ps = lonlat.shape[0]
    for start in range(0, n_ps, chunk_size):
        stop = min(start + chunk_size, n_ps)
        layer.StartTransaction()
        for i in range(start, stop):
            feature = ogr.Feature(feature_def)
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(float(lonlat[i, 0]), float(lonlat[i, 1]))
            feature.SetGeometry(point)
            feature.SetField("vel", float(velocity[i]))
            for col, label in enumerate(date_labels):
                feature.SetField("D" + label, float(ph_mm[i, col]))
            layer.CreateFeature(feature)
            feature = None
        layer.CommitTransaction()
        logger.info("Wrote vector features %d / %d", stop, n_ps)

    ds = None


def run_export(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    psver: Optional[int] = None,
    correction: str = "v-dso",
    output_format: str = "gpkg",
    layer_name: str = "stamps_points",
    deramp: bool = True,
    deramp_degree: Optional[float] = None,
    max_points: Optional[int] = None,
) -> None:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = StampsConfig(work_dir=input_path)
    if not cfg._loaded:
        cfg.load()

    v = psver if psver is not None else _get_psver(input_path, default=2)
    degree = deramp_degree if deramp_degree is not None else _load_deramp_degree(input_path)

    ph_uw, lonlat, xy, day, master_day, unwrap_ix, ramp = _load_corrected_phase(
        input_path,
        psver=v,
        correction=correction,
        apply_deramp=deramp,
        deramp_degree=degree,
    )

    if max_points is not None and max_points > 0:
        logger.warning("Debug export limited to first %d points", max_points)
        ph_uw = ph_uw[:max_points]
        lonlat = lonlat[:max_points]
        xy = xy[:max_points]

    wavelength_m = float(np.asarray(cfg.getparm("lambda")).ravel()[0])
    covariance = (
        None
        if str(cfg.getparm("small_baseline_flag") or "n").strip().lower() == "y"
        else _load_covariance(input_path, v, unwrap_ix)
    )
    logger.info("Computing velocity for %d PS and %d dates", ph_uw.shape[0], ph_uw.shape[1])
    velocity = _velocity_from_phase(ph_uw, day, master_day, wavelength_m, covariance)
    ph_mm, dates, date_labels, output_master_day = _build_full_time_series(
        ph_uw,
        day,
        master_day,
        wavelength_m,
        input_path,
    )

    out_correction = correction + ("o" if deramp and not correction.endswith("o") else "")
    _write_plot_h5(
        output_dir,
        velocity,
        ph_mm,
        lonlat,
        dates,
        date_labels,
        output_master_day,
        out_correction,
    )

    output_format = output_format.lower()
    if output_format in {"gpkg", "both"}:
        gpkg_path = output_dir / f"{input_path.name}_{out_correction}.gpkg"
        logger.info("Writing GeoPackage: %s", gpkg_path)
        _write_vector_ogr(gpkg_path, "GPKG", layer_name, lonlat, velocity, ph_mm, date_labels)
    if output_format in {"shp", "shapefile", "both"}:
        shp_path = output_dir / f"{input_path.name}_{out_correction}.shp"
        logger.info("Writing Shapefile: %s", shp_path)
        _write_vector_ogr(shp_path, "ESRI Shapefile", layer_name, lonlat, velocity, ph_mm, date_labels)
    if output_format not in {"gpkg", "shp", "shapefile", "both"}:
        raise ValueError("output_format must be gpkg, shp, shapefile, or both")


def main() -> int:
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="Export Python StaMPS velocity and displacement time series to GeoPackage/Shapefile.",
    )
    parser.add_argument("--input-path", "--input_path", required=True, help="Merged StaMPS project directory")
    parser.add_argument("--output-dir", "--output_dir", required=True, help="Output directory")
    parser.add_argument("--psver", type=int, default=None, help="PS version, default from psver.h5 or 2")
    parser.add_argument(
        "--correction",
        default="v-dso",
        choices=["v", "v-d", "v-do", "v-s", "v-so", "v-ds", "v-dso"],
        help="ps_plot velocity correction style. Default: v-dso.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="gpkg",
        choices=["gpkg", "shp", "shapefile", "both"],
        help="Vector output format. Default: gpkg.",
    )
    parser.add_argument("--layer-name", default="stamps_points", help="GeoPackage/Shapefile layer name")
    parser.add_argument("--no-deramp", action="store_true", help="Disable ps_deramp before export")
    parser.add_argument("--deramp-degree", type=float, default=None, help="Override deramp polynomial degree")
    parser.add_argument("--max-points", type=int, default=None, help="Debug/testing: export only first N points")

    args = parser.parse_args()
    try:
        run_export(
            input_path=args.input_path,
            output_dir=args.output_dir,
            psver=args.psver,
            correction=args.correction,
            output_format=args.output_format,
            layer_name=args.layer_name,
            deramp=not args.no_deramp,
            deramp_degree=args.deramp_degree,
            max_points=args.max_points,
        )
    except Exception:
        logger.exception("Export failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
