#!/usr/bin/env python3
"""
ps_merge_patches.py — Python port of ps_merge_patches.m + ps_calc_ifg_std.m
=============================================================================

Merge multiple patch directories into a single dataset and compute per-IFG
phase noise standard deviation.

MATLAB -> Python mapping
-----------------------
+----------------------------------------------+---------------------------------------------------+
| MATLAB concept                               | Python mapping                                    |
+==============================================+===================================================+
| ``ps_merge_patches(psver)``                  | ``PatchMerger(project_dir).run()``                |
+----------------------------------------------+---------------------------------------------------+
| ``patch_noover.in`` (non-overlap boundary)   | Loaded as 4-element array [rg1 rg2 az1 az2]      |
+----------------------------------------------+---------------------------------------------------+
| ``grid_size == 0`` (no resampling)           | Direct concatenation with overlap removal         |
+----------------------------------------------+---------------------------------------------------+
| ``grid_size ~= 0`` (resampling)              | Weighted grid-cell averaging (full implementation)|
+----------------------------------------------+---------------------------------------------------+
| ``llh2local(lonlat', ll0) * 1000``           | ``llh2local()`` (WGS84 projection, returns m)     |
+----------------------------------------------+---------------------------------------------------+
| Duplicate lon/lat removal (keep highest coh) | ``_remove_duplicates()``                          |
+----------------------------------------------+---------------------------------------------------+
| Heading-based rotation + y-sort              | ``_compute_xy_and_sort()``                        |
+----------------------------------------------+---------------------------------------------------+
| ``ps_calc_ifg_std``                          | ``calc_ifg_std()`` (standalone function)          |
+----------------------------------------------+---------------------------------------------------+

Step 5 flow (stamps.m)
-----------------------
**Part 5a — per-patch** (inside patch loop)::

    ps_correct_phase      -> rc2.h5 in each PATCH_*

**Part 5b — project root** (after patch loop)::

    ps_merge_patches      -> merged ps2, pm2, ph2, rc2, bp2, hgt2, inc2, la2
    ps_calc_ifg_std       -> ifgstd2.h5

Refactored from:
  - ps_merge_patches.m (Andy Hooper, September 2006)
  - ps_calc_ifg_std.m  (Andy Hooper, June 2006)
  - llh2local.m        (Peter Cervelli, September 2000)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig, load_mat  # noqa: E402

logger = logging.getLogger("stamps")


def _as_matlab_index_column(values: np.ndarray) -> np.ndarray:
    """Return an int32 column vector using MATLAB 1-based indexing."""
    arr = np.asarray(values).ravel().astype(np.int32)
    if arr.size > 0 and arr.min() == 0:
        arr = arr + 1
    return arr[:, None]


# ============================================================================
# llh2local — WGS84 lon/lat to local XY (metres)
# ============================================================================

def llh2local(lonlat: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to local XY in **metres**.

    Parameters
    ----------
    lonlat : (n, 2) array — columns are [lon, lat]
    origin : (2,) array  — [lon0, lat0] in degrees

    Returns
    -------
    xy : (n, 2) array — columns are [x, y] in metres

    MATLAB correspondence (llh2local.m)
    ------------------------------------
    The MATLAB version accepts ``llh`` as (2, n) with rows [lon; lat]
    and returns ``xy`` as (2, n) with rows [x; y], in **kilometres**.
    Here we use (n, 2) row-oriented layout and return **metres** to align
    with ``ps_merge_patches.m`` which multiplies by 1000 immediately::

        xy = llh2local(lonlat', ll0) * 1000;

    WGS84 constants:
        a = 6378137.0        (semi-major axis, m)
        e = 0.08209443794970 (first eccentricity)
    """
    a = 6378137.0
    e = 0.08209443794970

    lon_rad = np.deg2rad(lonlat[:, 0].astype(np.float64))
    lat_rad = np.deg2rad(lonlat[:, 1].astype(np.float64))
    lon0 = np.deg2rad(float(origin[0]))
    lat0 = np.deg2rad(float(origin[1]))

    xy = np.zeros((len(lonlat), 2), dtype=np.float64)

    # Non-zero latitude
    nz = lat_rad != 0
    if nz.any():
        dlambda = lon_rad[nz] - lon0
        M = a * (
            (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat_rad[nz]
            - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat_rad[nz])
            + (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat_rad[nz])
            - (35*e**6/3072) * np.sin(6*lat_rad[nz])
        )
        M0 = a * (
            (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat0
            - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat0)
            + (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat0)
            - (35*e**6/3072) * np.sin(6*lat0)
        )
        N = a / np.sqrt(1 - e**2 * np.sin(lat_rad[nz])**2)
        E_angle = dlambda * np.sin(lat_rad[nz])

        # cot(lat) = cos(lat)/sin(lat)
        cot_lat = np.cos(lat_rad[nz]) / np.sin(lat_rad[nz])
        xy[nz, 0] = N * cot_lat * np.sin(E_angle)
        xy[nz, 1] = M - M0 + N * cot_lat * (1 - np.cos(E_angle))

    # Zero latitude
    z = ~nz
    if z.any():
        dlambda = lon_rad[z] - lon0
        M0 = a * (
            (1 - e**2/4 - 3*e**4/64 - 5*e**6/256) * lat0
            - (3*e**2/8 + 3*e**4/32 + 45*e**6/1024) * np.sin(2*lat0)
            + (15*e**4/256 + 45*e**6/1024) * np.sin(4*lat0)
            - (35*e**6/3072) * np.sin(6*lat0)
        )
        xy[z, 0] = a * dlambda
        xy[z, 1] = -M0

    return xy  # metres


# ============================================================================
# HDF5/MAT data loading helpers
# ============================================================================

def _load_h5_or_mat(directory: Path, basename: str, keys: list) -> dict:
    """Load datasets from HDF5 or MATLAB .mat file."""
    h5_path = directory / (basename + ".h5")
    mat_path = directory / (basename + ".mat")

    data: dict = {}
    if h5_path.is_file():
        with h5py.File(str(h5_path), "r") as hf:
            for k in keys:
                if k in hf:
                    data[k] = np.asarray(hf[k][:])
        return data
    elif mat_path.is_file():
        d = load_mat(mat_path)
        for k in keys:
            if k in d:
                data[k] = np.asarray(d[k])
        return data
    else:
        raise FileNotFoundError(
            f"Neither {h5_path.name} nor {mat_path.name} found in {directory}"
        )


def _load_optional(directory: Path, basename: str, key: str) -> Optional[np.ndarray]:
    """Try to load a single variable; return None if file missing."""
    for ext in (".h5", ".mat"):
        path = directory / (basename + ext)
        if path.is_file():
            if ext == ".h5":
                with h5py.File(str(path), "r") as hf:
                    if key in hf:
                        return np.asarray(hf[key][:])
            else:
                d = load_mat(path)
                if key in d:
                    return np.asarray(d[key])
    return None


def _load_ph(directory: Path, ver: int) -> Optional[np.ndarray]:
    """Load complex phase from ph<ver> or ps<ver>."""
    for src in [f"ph{ver}", f"ps{ver}"]:
        for ext in (".h5", ".mat"):
            path = directory / (src + ext)
            if path.is_file():
                if ext == ".h5":
                    with h5py.File(str(path), "r") as hf:
                        if "ph" in hf:
                            return np.asarray(hf["ph"][:])
                else:
                    d = load_mat(path)
                    if "ph" in d:
                        return np.asarray(d["ph"])
    return None


# ============================================================================
# HDF5 persistence helpers
# ============================================================================

def _save_merged_h5(
    h5_path: Path, datasets: Dict[str, np.ndarray],
    compress_keys: Optional[set] = None,
) -> None:
    """Save a dictionary of arrays to an HDF5 file.

    Datasets whose keys are in *compress_keys* get gzip compression.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    if compress_keys is None:
        compress_keys = set()
    with h5py.File(str(h5_path), "w") as hf:
        for k, v in datasets.items():
            comp = "gzip" if k in compress_keys else None
            hf.create_dataset(k, data=v, compression=comp)
    logger.info("Saved %s", h5_path.name)


# ============================================================================
# ps_calc_ifg_std — per-IFG phase noise standard deviation
# ============================================================================

def calc_ifg_std(
    project_dir: Path,
    psver: int = 2,
) -> np.ndarray:
    """Calculate per-IFG phase noise standard deviation (degrees).

    MATLAB correspondence (ps_calc_ifg_std.m)
    -------------------------------------------
    ::

        % SB mode:
        ph_diff = angle(ph .* conj(pm.ph_patch)
                        .* exp(-j*(K_ps .* bperp_mat)));
        % PS mode:
        bperp_mat_full = [bperp_mat(:,1:mi-1), zeros, bperp_mat(:,mi:end)]
        ph_patch_full  = [ph_patch(:,1:mi-1),   ones,  ph_patch(:,mi:end)]
        ph_diff = angle(ph .* conj(ph_patch_full)
                        .* exp(-j*(K_ps .* bperp_mat_full + C_ps)));

        ifg_std = sqrt(sum(ph_diff.^2) / n_ps) * 180/pi

    Returns
    -------
    ifg_std : (n_ifg,) float32 — noise std in degrees for each IFG.
    """
    logger.info("Estimating noise standard deviation (degrees)...")
    cfg = StampsConfig(work_dir=project_dir)
    if not cfg._loaded:
        cfg.load()
    small_baseline_flag = str(cfg.getparm("small_baseline_flag")).strip().lower()

    v = psver
    ps = _load_h5_or_mat(project_dir, f"ps{v}",
                         ["n_ps", "n_ifg", "master_day", "day", "master_ix",
                          "xy", "ifgday"])
    n_ps = int(np.asarray(ps["n_ps"]).ravel()[0])
    n_ifg = int(np.asarray(ps["n_ifg"]).ravel()[0])
    master_ix_1 = int(np.asarray(ps["master_ix"]).ravel()[0])
    master_ix_0 = master_ix_1 - 1

    pm = _load_h5_or_mat(project_dir, f"pm{v}",
                         ["K_ps", "C_ps", "ph_patch"])
    K_ps = np.asarray(pm["K_ps"], dtype=np.float32).ravel()
    C_ps = np.asarray(pm["C_ps"], dtype=np.float32).ravel()
    ph_patch = np.asarray(pm["ph_patch"])

    bp = _load_h5_or_mat(project_dir, f"bp{v}", ["bperp_mat"])
    bperp_mat = np.asarray(bp["bperp_mat"], dtype=np.float32)

    ph = _load_ph(project_dir, v)
    if ph is None:
        raise FileNotFoundError(f"ph data not found for version {v}")

    # --- Compute ph_diff (vectorised) ---
    if small_baseline_flag == "y":
        # MATLAB (line 38):
        #   ph_diff = angle(ph .* conj(ph_patch) .* exp(-j*(K_ps.*bperp_mat)))
        ph_diff = np.angle(
            ph * np.conj(ph_patch)
            * np.exp(-1j * (K_ps[:, None] * bperp_mat))
        )
    else:
        # PS mode: insert zero/ones column at master position
        bperp_mat_full = np.insert(bperp_mat, master_ix_0, 0.0, axis=1).astype(np.float32)
        ph_patch_full = np.insert(
            ph_patch, master_ix_0,
            np.ones(n_ps, dtype=ph_patch.dtype), axis=1
        )
        ph_diff = np.angle(
            ph * np.conj(ph_patch_full)
            * np.exp(-1j * (K_ps[:, None] * bperp_mat_full + C_ps[:, None]))
        )

    # MATLAB (line 45): ifg_std = sqrt(sum(ph_diff.^2)/n_ps) * 180/pi
    # sum along axis=0 (over pixels), then divide by n_ps
    ifg_std = (np.sqrt(np.sum(ph_diff ** 2, axis=0) / n_ps) * 180.0 / np.pi).astype(np.float32)

    # Log a few values
    logger.info("  IFG std range: %.2f - %.2f degrees (mean=%.2f)",
                ifg_std.min(), ifg_std.max(), ifg_std.mean())

    # Save — MATLAB: ifg_std is (n_ifg, 1) column vector
    ifgstd_path = project_dir / f"ifgstd{v}.h5"
    _save_merged_h5(ifgstd_path, {"ifg_std": ifg_std.ravel()[:, None]})

    return ifg_std


# ============================================================================
# PatchMerger
# ============================================================================

class PatchMerger:
    """Merge multiple patches into a single dataset.

    MATLAB correspondence: ``ps_merge_patches.m``

    Parameters
    ----------
    project_dir : Path
        Top-level project directory containing ``patch.list`` or ``PATCH_*``.
    psver : int
        PS version number (default 2).
    """

    def __init__(
        self,
        project_dir: Union[str, Path],
        psver: int = 2,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.psver = psver
        self._cfg = StampsConfig(work_dir=self.project_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Execute patch merging.

        Returns
        -------
        n_ps : int — total number of merged PS.
        """
        t0 = time.time()
        logger.info("Merging patches...")

        v = self.psver
        small_baseline_flag = str(
            self._cfg.getparm("small_baseline_flag")
        ).strip().lower()
        grid_size = int(self._cfg.getparm("merge_resample_size"))
        merge_stdev = float(self._cfg.getparm("merge_standard_dev"))

        heading_raw = self._cfg.getparm("heading")
        heading = float(heading_raw) if heading_raw is not None else 0.0

        # MATLAB (line 36-39): phase_accuracy, min_weight, max_coh
        phase_accuracy = 10.0 * np.pi / 180.0
        min_weight = 1.0 / merge_stdev ** 2
        rng = np.random.RandomState(1001)
        max_coh = float(np.abs(np.sum(np.exp(1j * rng.randn(1000) * phase_accuracy))) / 1000)

        # Discover patches
        patch_dirs = self._discover_patch_dirs()
        n_patch = len(patch_dirs)
        logger.info("  %d patch(es) to merge, grid_size=%d", n_patch, grid_size)

        # ---- Accumulators ----
        all_ij: List[np.ndarray] = []
        all_lonlat: List[np.ndarray] = []
        all_ph: List[np.ndarray] = []
        all_ph_rc: List[np.ndarray] = []
        all_ph_reref: List[np.ndarray] = []
        all_ph_patch: List[np.ndarray] = []
        all_ph_res: List[np.ndarray] = []
        all_K_ps: List[np.ndarray] = []
        all_C_ps: List[np.ndarray] = []
        all_coh_ps: List[np.ndarray] = []
        all_bperp_mat: List[np.ndarray] = []
        all_la: List[np.ndarray] = []
        all_inc: List[np.ndarray] = []
        all_hgt: List[np.ndarray] = []
        remove_ix: List[int] = []  # indices to remove from accumulated arrays

        # Reference ps structure from last patch (for metadata)
        ps_ref: dict = {}
        n_ifg = 0
        n_image = 0
        running_total = 0  # cumulative PS count for offset

        for i_p, pdir in enumerate(patch_dirs):
            logger.info("  Processing %s", pdir.name)

            # Load ps<ver>
            ps = _load_h5_or_mat(pdir, f"ps{v}",
                                 ["n_ps", "n_ifg", "n_image", "ij", "lonlat",
                                  "xy", "master_day", "day", "master_ix",
                                  "bperp", "day_ix", "ifgday", "ifgday_ix", "ll0"])
            n_ps_patch = int(np.asarray(ps["n_ps"]).ravel()[0])
            n_ifg = int(np.asarray(ps["n_ifg"]).ravel()[0])
            n_image = int(np.asarray(ps.get("n_image", ps["n_ifg"])).ravel()[0])
            ps_ref = ps

            ij_patch = np.asarray(ps["ij"])     # (n_ps, 3) — col0=idx, col1=az, col2=rg
            lonlat_patch = np.asarray(ps["lonlat"])
            xy_patch = np.asarray(ps.get("xy", np.zeros((n_ps_patch, 3))))

            # Load patch_noover.in to define non-overlap boundary
            # MATLAB format: [rg_start, rg_end, az_start, az_end]
            noover = self._load_patch_noover(pdir)
            if noover is not None:
                rg1, rg2, az1, az2 = noover
                # MATLAB (line 107): ix = ij(:,2)>=az_start-1 & ... & ij(:,3)>=rg_start-1 ...
                # ij columns: col1=az(row), col2=rg(col) — 1-based in MATLAB
                az_vals = ij_patch[:, 1]
                rg_vals = ij_patch[:, 2]
                ix = (
                    (az_vals >= az1 - 1) & (az_vals <= az2 - 1)
                    & (rg_vals >= rg1 - 1) & (rg_vals <= rg2 - 1)
                )
            else:
                ix = np.ones(n_ps_patch, dtype=bool)

            if ix.sum() == 0:
                logger.warning("  No PS in non-overlap region for %s", pdir.name)
                continue

            # --- grid_size == 0: no resampling, direct concatenation ---
            if grid_size == 0:
                # Find overlap with already-accumulated ij
                if len(all_ij) > 0:
                    prev_ij = np.vstack(all_ij)
                    patch_ij_noover = ij_patch[ix, 1:3]
                    # intersect: rows in patch_noover that also appear in prev_ij
                    # MATLAB (line 115-120)
                    _, IA, IB = _intersect_rows(patch_ij_noover, prev_ij)
                    remove_ix.extend(IB.tolist())

                    # Also find all patch pixels that overlap with previous
                    _, IA_all, _ = _intersect_rows(ij_patch[:, 1:3], prev_ij)
                    ix_ex = np.ones(n_ps_patch, dtype=bool)
                    ix_ex[IA_all] = False
                    ix[ix_ex] = True  # keep noover + exclusive non-intersecting

                all_ij.append(ij_patch[ix, 1:3])
                all_lonlat.append(lonlat_patch[ix])

                # ph
                ph_w = _load_ph(pdir, v)
                if ph_w is not None:
                    all_ph.append(ph_w[ix])

                # rc
                rc = _load_h5_or_mat(pdir, f"rc{v}", ["ph_rc", "ph_reref"])
                all_ph_rc.append(np.asarray(rc["ph_rc"])[ix])
                if "ph_reref" in rc and small_baseline_flag != "y":
                    all_ph_reref.append(np.asarray(rc["ph_reref"])[ix])

                # pm
                pm = _load_h5_or_mat(pdir, f"pm{v}",
                                     ["ph_patch", "ph_res", "K_ps", "C_ps", "coh_ps"])
                all_ph_patch.append(np.asarray(pm["ph_patch"])[ix])
                if "ph_res" in pm:
                    ph_res_arr = np.asarray(pm["ph_res"])
                    if ph_res_arr.ndim >= 2 and ph_res_arr.shape[0] == n_ps_patch:
                        all_ph_res.append(ph_res_arr[ix])
                if "K_ps" in pm:
                    all_K_ps.append(np.asarray(pm["K_ps"]).ravel()[ix, None])
                if "C_ps" in pm:
                    all_C_ps.append(np.asarray(pm["C_ps"]).ravel()[ix, None])
                if "coh_ps" in pm:
                    all_coh_ps.append(np.asarray(pm["coh_ps"]).ravel()[ix, None])

                # bp
                bp = _load_h5_or_mat(pdir, f"bp{v}", ["bperp_mat"])
                all_bperp_mat.append(np.asarray(bp["bperp_mat"])[ix])

                # Optional: la, inc, hgt
                la_arr = _load_optional(pdir, f"la{v}", "la")
                if la_arr is not None:
                    all_la.append(la_arr.ravel()[ix, None])
                inc_arr = _load_optional(pdir, f"inc{v}", "inc")
                if inc_arr is not None:
                    all_inc.append(inc_arr.ravel()[ix, None])
                hgt_arr = _load_optional(pdir, f"hgt{v}", "hgt")
                if hgt_arr is not None:
                    all_hgt.append(hgt_arr.ravel()[ix, None])

                running_total += int(ix.sum())

            else:
                # --- grid_size > 0: weighted resampling (full implementation) ---
                self._merge_resampled(
                    pdir, ps, ix, v, n_ifg, grid_size,
                    phase_accuracy, min_weight, max_coh,
                    small_baseline_flag,
                    all_ij, all_lonlat, all_ph, all_ph_rc, all_ph_reref,
                    all_ph_patch, all_ph_res, all_K_ps, all_C_ps, all_coh_ps,
                    all_bperp_mat, all_la, all_inc, all_hgt,
                )

        # ---- Concatenate ----
        def _vstack_safe(arrs: List[np.ndarray]) -> np.ndarray:
            if not arrs:
                return np.empty((0, 0))
            return np.vstack(arrs)

        ij = _vstack_safe(all_ij)               # (N, 2)
        lonlat = _vstack_safe(all_lonlat)       # (N, 2)
        coh_ps = _vstack_safe(all_coh_ps).ravel() if all_coh_ps else np.empty(0)

        n_ps_orig = ij.shape[0]
        keep_ix = np.ones(n_ps_orig, dtype=bool)

        # Remove overlap pixels (grid_size==0 only)
        if remove_ix:
            keep_ix[np.array(remove_ix, dtype=np.int64)] = False

        # ---- Remove duplicate lonlat (MATLAB lines 464-481) ----
        keep_ix = self._remove_duplicates(lonlat, coh_ps, keep_ix)
        lonlat = lonlat[keep_ix]

        # ---- Compute ll0, xy, heading rotation, sort (MATLAB lines 484-524) ----
        ll0 = (lonlat.max(axis=0) + lonlat.min(axis=0)) / 2.0
        xy_m = llh2local(lonlat, ll0)  # metres
        xy_m = self._apply_heading_rotation(xy_m, heading)
        xy_m = np.round(xy_m, 3)       # round to mm (MATLAB line 523)
        xy_m = xy_m.astype(np.float32)

        # Sort by (y, x) ascending — MATLAB (line 516)
        sort_ix = np.lexsort((xy_m[:, 0], xy_m[:, 1]))

        # Build xy with index column: [idx, x, y] — MATLAB (line 522)
        xy_sorted = np.column_stack([
            np.arange(1, len(sort_ix) + 1, dtype=np.float32),
            xy_m[sort_ix],
        ])
        lonlat_sorted = lonlat[sort_ix]

        # Map sort_ix back through keep_ix to the original concatenated arrays
        all_ix = np.arange(n_ps_orig)
        keep_ix_num = all_ix[keep_ix]
        final_ix = keep_ix_num[sort_ix]

        n_ps = len(final_ix)
        logger.info("  Writing merged dataset (%d pixels)", n_ps)

        # ---- Gather and sort concatenated arrays ----
        ij_merged = ij[final_ix]

        ph_rc_all = _vstack_safe(all_ph_rc)[final_ix] if all_ph_rc else np.empty((0, 0))
        # Normalise ph_rc (MATLAB line 536): ph_rc(ph_rc~=0) = ph_rc./abs(ph_rc)
        nz = ph_rc_all != 0
        ph_rc_all[nz] = ph_rc_all[nz] / np.abs(ph_rc_all[nz])

        ph_reref_all = _vstack_safe(all_ph_reref)[final_ix] if all_ph_reref else None

        ph_patch_all = _vstack_safe(all_ph_patch)[final_ix] if all_ph_patch else np.empty((0, 0))
        ph_res_all = _vstack_safe(all_ph_res)[final_ix] if all_ph_res else np.empty((0, 0))
        K_ps_all = _vstack_safe(all_K_ps)[final_ix] if all_K_ps else np.empty((0, 0))
        C_ps_all = _vstack_safe(all_C_ps)[final_ix] if all_C_ps else np.empty((0, 0))
        coh_ps_all = _vstack_safe(all_coh_ps)[final_ix] if all_coh_ps else np.empty((0, 0))
        bperp_mat_all = _vstack_safe(all_bperp_mat)[final_ix] if all_bperp_mat else np.empty((0, 0))
        ph_all = _vstack_safe(all_ph)[final_ix] if all_ph else np.empty((0, 0))

        def _optional_merged(arrs: List[np.ndarray], name: str) -> Optional[np.ndarray]:
            if not arrs:
                return None
            arr = _vstack_safe(arrs)
            if arr.shape[0] != n_ps_orig:
                logger.warning(
                    "Skipping merged %s: optional rows %d do not match merged source rows %d",
                    name,
                    arr.shape[0],
                    n_ps_orig,
                )
                return None
            return arr[final_ix]

        la_all = _optional_merged(all_la, "la")
        inc_all = _optional_merged(all_inc, "inc")
        hgt_all = _optional_merged(all_hgt, "hgt")

        # ---- Save merged files at project root (MATLAB lines 534-643) ----
        out = self.project_dir
        compress = {"ph_rc", "ph_reref", "ph", "ph_patch", "ph_res", "bperp_mat"}

        # rc2
        rc_data: Dict[str, np.ndarray] = {"ph_rc": ph_rc_all.astype(np.complex64)}
        if ph_reref_all is not None:
            rc_data["ph_reref"] = ph_reref_all.astype(np.complex64)
        _save_merged_h5(out / f"rc{v}.h5", rc_data, compress)

        # pm2
        pm_data: Dict[str, np.ndarray] = {"ph_patch": ph_patch_all.astype(np.complex64)}
        if ph_res_all.size > 0:
            pm_data["ph_res"] = ph_res_all.astype(np.float32)
        if K_ps_all.size > 0:
            pm_data["K_ps"] = K_ps_all.astype(np.float64)
        if C_ps_all.size > 0:
            pm_data["C_ps"] = C_ps_all.astype(np.float64)
        if coh_ps_all.size > 0:
            pm_data["coh_ps"] = coh_ps_all.astype(np.float64)
        _save_merged_h5(out / f"pm{v}.h5", pm_data, compress)

        # ph2
        if ph_all.size > 0:
            _save_merged_h5(out / f"ph{v}.h5",
                            {"ph": ph_all.astype(np.complex64)}, compress)

        # bp2
        if bperp_mat_all.size > 0:
            _save_merged_h5(out / f"bp{v}.h5",
                            {"bperp_mat": bperp_mat_all.astype(np.float32)}, compress)

        # la2, inc2, hgt2
        if la_all is not None:
            _save_merged_h5(out / f"la{v}.h5", {"la": la_all.astype(np.float64)})
        if inc_all is not None:
            _save_merged_h5(out / f"inc{v}.h5", {"inc": inc_all.astype(np.float64)})
        if hgt_all is not None:
            _save_merged_h5(out / f"hgt{v}.h5", {"hgt": hgt_all.astype(np.float64)})

        # ps2 — the main metadata file
        ps_out: Dict[str, np.ndarray] = {
            "n_ps": np.array([[n_ps]], dtype=np.int32),
            "n_ifg": np.array([[n_ifg]], dtype=np.int32),
            "n_image": np.array([[n_image]], dtype=np.int32),
            "ij": np.column_stack([
                np.arange(1, n_ps + 1, dtype=np.int32),
                ij_merged,
            ]).astype(np.int32),
            "xy": xy_sorted,
            "lonlat": lonlat_sorted,
            "ll0": ll0[None, :],  # (1, 2)
        }
        # Copy scalar fields from reference patch
        for k in ["bperp", "day", "day_ix", "master_day", "master_ix",
                   "ifgday", "ifgday_ix"]:
            if k in ps_ref:
                val = np.asarray(ps_ref[k])
                if k == "day_ix":
                    ps_out[k] = _as_matlab_index_column(val)
                    continue
                if val.ndim == 0:
                    val = val.reshape(1, 1)
                elif val.ndim == 1:
                    val = val[:, None]
                ps_out[k] = val
        _save_merged_h5(out / f"ps{v}.h5", ps_out)

        # psver
        with h5py.File(str(out / "psver.h5"), "w") as hf:
            hf.create_dataset("psver", data=np.array([[v]], dtype=np.int32))

        elapsed = time.time() - t0
        logger.info("Merge complete: %d PS (%.1f s)", n_ps, elapsed)
        return n_ps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_patch_dirs(self) -> List[Path]:
        """Find patch directories from patch.list or PATCH_* glob."""
        patch_list = self.project_dir / "patch.list"
        dirs: List[Path] = []
        if patch_list.is_file():
            for line in patch_list.read_text(encoding="utf-8").splitlines():
                name = line.strip()
                if name:
                    candidate = self.project_dir / name
                    if candidate.is_dir():
                        dirs.append(candidate)
        else:
            dirs = sorted(
                p for p in self.project_dir.iterdir()
                if p.is_dir() and p.name.upper().startswith("PATCH_")
            )
        return dirs

    def _load_patch_noover(self, patch_dir: Path) -> Optional[Tuple[int, int, int, int]]:
        """Load ``patch_noover.in`` boundary file.

        Returns (rg_start, rg_end, az_start, az_end) or None.
        MATLAB format: 4 numbers, one per line.
        """
        path = patch_dir / "patch_noover.in"
        if not path.is_file():
            return None
        vals = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                vals.append(int(line))
        if len(vals) >= 4:
            return (vals[0], vals[1], vals[2], vals[3])
        return None

    @staticmethod
    def _remove_duplicates(
        lonlat: np.ndarray,
        coh_ps: np.ndarray,
        keep_ix: np.ndarray,
    ) -> np.ndarray:
        """Remove pixels with duplicate lon/lat, keeping highest coherence.

        MATLAB correspondence (ps_merge_patches.m lines 464-481).
        """
        if keep_ix.sum() == 0:
            return keep_ix

        lonlat_kept = lonlat[keep_ix]
        coh_kept = coh_ps[keep_ix] if len(coh_ps) > 0 else np.zeros(keep_ix.sum())
        keep_ix_num = np.where(keep_ix)[0]

        _, unique_idx = np.unique(lonlat_kept, axis=0, return_index=True)
        all_idx = np.arange(len(lonlat_kept))
        dup_idx = np.setdiff1d(all_idx, unique_idx)

        n_dropped = 0
        for di in dup_idx:
            same = np.where(
                (lonlat_kept[:, 0] == lonlat_kept[di, 0])
                & (lonlat_kept[:, 1] == lonlat_kept[di, 1])
            )[0]
            orig_idx = keep_ix_num[same]
            best = orig_idx[np.argmax(coh_kept[same])]
            for oi in orig_idx:
                if oi != best and keep_ix[oi]:
                    keep_ix[oi] = False
                    n_dropped += 1

        if n_dropped > 0:
            logger.info("  %d pixels with duplicate lon/lat dropped", n_dropped)
        return keep_ix

    @staticmethod
    def _apply_heading_rotation(xy: np.ndarray, heading: float) -> np.ndarray:
        """Rotate XY to align scene axes, checking if rotation improves span.

        MATLAB correspondence (ps_merge_patches.m lines 495-512)::

            theta = (180 - heading) * pi/180;
            rotm = [cos(theta) sin(theta); -sin(theta) cos(theta)];
            xynew = rotm * xy';
            if range(xynew) < range(xy)  -> use rotated
        """
        theta = (180.0 - heading) * np.pi / 180.0
        if theta > np.pi:
            theta -= 2 * np.pi

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotm = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        xynew = (rotm @ xy.T).T  # (n, 2)

        # Check if rotation reduces both x and y spans
        x_span_old = xy[:, 0].max() - xy[:, 0].min()
        y_span_old = xy[:, 1].max() - xy[:, 1].min()
        x_span_new = xynew[:, 0].max() - xynew[:, 0].min()
        y_span_new = xynew[:, 1].max() - xynew[:, 1].min()

        if x_span_new < x_span_old and y_span_new < y_span_old:
            logger.info("  Rotating xy by %.2f degrees", theta * 180.0 / np.pi)
            return xynew
        return xy

    def _merge_resampled(
        self,
        pdir, ps, ix, v, n_ifg, grid_size,
        phase_accuracy, min_weight, max_coh,
        small_baseline_flag,
        all_ij, all_lonlat, all_ph, all_ph_rc, all_ph_reref,
        all_ph_patch, all_ph_res, all_K_ps, all_C_ps, all_coh_ps,
        all_bperp_mat, all_la, all_inc, all_hgt,
    ) -> None:
        """Weighted resampling merge for grid_size > 0.

        MATLAB correspondence (ps_merge_patches.m lines 121-171, 176-389).

        This handles the case where multiple PS in the same grid cell
        are weighted-averaged into a single representative point.
        """
        n_ps_patch = int(np.asarray(ps["n_ps"]).ravel()[0])
        xy_patch = np.asarray(ps.get("xy", np.zeros((n_ps_patch, 3))), dtype=np.float32)
        ij_patch = np.asarray(ps["ij"])
        lonlat_patch = np.asarray(ps["lonlat"])

        # Grid cell assignment (MATLAB lines 122-131)
        ix_idx = np.where(ix)[0]
        xy_sel = xy_patch[ix_idx]
        xy_min = xy_sel.min(axis=0)

        g_ij = np.column_stack([
            np.ceil((xy_sel[:, 2] - xy_min[2] + 1e-9) / grid_size).astype(int),
            np.ceil((xy_sel[:, 1] - xy_min[1] + 1e-9) / grid_size).astype(int),
        ])

        # Unique grid cells and sort
        _, first_idx, g_ix = np.unique(g_ij, axis=0, return_index=True, return_inverse=True)
        sort_order = np.argsort(g_ix)
        g_ix = g_ix[sort_order]
        ix_idx = ix_idx[sort_order]

        # Load pm for weight computation
        pm = _load_h5_or_mat(pdir, f"pm{v}", ["ph_res", "coh_ps", "C_ps"])
        ph_res_raw = np.asarray(pm["ph_res"]) if "ph_res" in pm else np.empty(0)
        C_ps_raw = np.asarray(pm["C_ps"]).ravel()
        coh_pm = np.asarray(pm["coh_ps"]).ravel() if "coh_ps" in pm else np.empty(0)

        # Centralise ph_res about zero (MATLAB line 133). Some lightweight
        # Step-4 outputs intentionally keep ph_res empty; in that case use the
        # saved coherence and a conservative phase-accuracy variance for merge
        # weights instead of manufacturing NaN weights from empty residuals.
        use_ph_res = (
            ph_res_raw.ndim >= 2
            and ph_res_raw.shape[0] == n_ps_patch
            and ph_res_raw.shape[1] > 0
        )
        if use_ph_res:
            ph_res_ctr = np.angle(np.exp(1j * (ph_res_raw - C_ps_raw[:, None])))
            if small_baseline_flag != "y":
                ph_res_ctr = np.column_stack([ph_res_ctr, C_ps_raw])
            sigsq_noise = np.var(ph_res_ctr, axis=1, ddof=1)
            coh_ps_all = np.abs(np.sum(np.exp(1j * ph_res_ctr), axis=1)) / n_ifg
        else:
            sigsq_noise = np.full(n_ps_patch, phase_accuracy ** 2, dtype=np.float64)
            if coh_pm.shape[0] == n_ps_patch:
                coh_ps_all = coh_pm.astype(np.float64, copy=False)
            else:
                coh_ps_all = np.zeros(n_ps_patch, dtype=np.float64)
        coh_ps_all = np.clip(coh_ps_all, 0.0, min(float(max_coh), 1.0 - 1e-6))
        sigsq_noise = np.nan_to_num(sigsq_noise, nan=phase_accuracy ** 2, posinf=phase_accuracy ** 2)
        sigsq_noise = np.maximum(sigsq_noise, phase_accuracy ** 2)

        ps_weight = 1.0 / sigsq_noise[ix_idx]
        ps_snr = 1.0 / (1.0 / coh_ps_all[ix_idx] ** 2 - 1.0)
        ps_snr = np.nan_to_num(ps_snr, nan=0.0, posinf=1e6, neginf=0.0)

        # Group boundaries
        changes = np.where(np.diff(g_ix))[0]
        l_ix = np.append(changes, len(g_ix) - 1)
        f_ix = np.concatenate([[0], l_ix[:-1] + 1])
        n_ps_g = len(f_ix)

        # Weight threshold check
        for gi in range(n_ps_g):
            sl = slice(f_ix[gi], l_ix[gi] + 1)
            ws = ps_weight[sl].sum()
            if (not np.isfinite(ws)) or ws <= 0 or ws < min_weight:
                ix_idx[sl] = -1  # mark for removal

        valid = (
            (ix_idx >= 0)
            & np.isfinite(ps_weight)
            & (ps_weight > 0)
            & np.isfinite(ps_snr)
            & (ps_snr >= 0)
        )
        g_ix = g_ix[valid]
        ps_weight = ps_weight[valid]
        ps_snr = ps_snr[valid]
        ix_idx = ix_idx[valid]

        if len(g_ix) == 0:
            return

        # Recompute group boundaries
        changes = np.where(np.diff(g_ix))[0]
        l_ix = np.append(changes, len(g_ix) - 1)
        f_ix = np.concatenate([[0], l_ix[:-1] + 1])
        n_ps_g = len(f_ix)

        # Weighted average for ij, lonlat
        ij_g = np.zeros((n_ps_g, 2), dtype=np.float64)
        lonlat_g = np.zeros((n_ps_g, 2), dtype=np.float64)
        for gi in range(n_ps_g):
            sl = slice(f_ix[gi], l_ix[gi] + 1)
            w = ps_weight[sl]
            ws = w.sum()
            if (not np.isfinite(ws)) or ws <= 0:
                continue
            ij_g[gi] = np.round(np.sum(ij_patch[ix_idx[sl], 1:3] * w[:, None], axis=0) / ws)
            lonlat_g[gi] = np.sum(lonlat_patch[ix_idx[sl]] * w[:, None], axis=0) / ws

        all_ij.append(ij_g.astype(np.int32))
        all_lonlat.append(lonlat_g)

        # Weighted averages for phase arrays, bperp, etc.
        self._merge_resampled_arrays(
            pdir, v, n_ifg, n_ps_g, ix_idx, f_ix, l_ix,
            ps_weight, ps_snr, small_baseline_flag,
            all_ph, all_ph_rc, all_ph_reref,
            all_ph_patch, all_ph_res, all_K_ps, all_C_ps, all_coh_ps,
            all_bperp_mat, all_la, all_inc, all_hgt,
        )

    def _merge_resampled_arrays(
        self, pdir, v, n_ifg, n_ps_g, ix_idx, f_ix, l_ix,
        ps_weight, ps_snr, small_baseline_flag,
        all_ph, all_ph_rc, all_ph_reref,
        all_ph_patch, all_ph_res, all_K_ps, all_C_ps, all_coh_ps,
        all_bperp_mat, all_la, all_inc, all_hgt,
    ):
        """Weighted-average all data arrays for resampled merge."""
        # ph
        ph_w = _load_ph(pdir, v)
        if ph_w is not None:
            ph_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex128)
            for gi in range(n_ps_g):
                sl = slice(f_ix[gi], l_ix[gi] + 1)
                w = ps_snr[sl]
                ph_g[gi] = np.sum(ph_w[ix_idx[sl]] * w[:, None], axis=0)
            all_ph.append(ph_g.astype(np.complex64))

        # rc
        rc = _load_h5_or_mat(pdir, f"rc{v}", ["ph_rc", "ph_reref"])
        rc_data = np.asarray(rc["ph_rc"])
        ph_rc_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex128)
        ph_reref_g = None
        if "ph_reref" in rc and small_baseline_flag != "y":
            ph_reref_g = np.zeros((n_ps_g, n_ifg), dtype=np.complex128)
        for gi in range(n_ps_g):
            sl = slice(f_ix[gi], l_ix[gi] + 1)
            w = ps_snr[sl]
            ph_rc_g[gi] = np.sum(rc_data[ix_idx[sl]] * w[:, None], axis=0)
            if ph_reref_g is not None:
                ph_reref_g[gi] = np.sum(
                    np.asarray(rc["ph_reref"])[ix_idx[sl]] * w[:, None], axis=0
                )
        all_ph_rc.append(ph_rc_g.astype(np.complex64))
        if ph_reref_g is not None:
            all_ph_reref.append(ph_reref_g.astype(np.complex64))

        # pm
        pm = _load_h5_or_mat(pdir, f"pm{v}",
                             ["ph_patch", "ph_res", "K_ps", "C_ps", "coh_ps"])
        pp_data = np.asarray(pm["ph_patch"])
        pp_ncol = pp_data.shape[1]
        pp_g = np.zeros((n_ps_g, pp_ncol), dtype=np.complex128)
        pr_raw = np.asarray(pm["ph_res"]) if "ph_res" in pm else None
        if pr_raw is not None and (
            pr_raw.ndim < 2 or pr_raw.shape[0] <= int(ix_idx.max())
        ):
            pr_raw = None
        pr_g = np.zeros((n_ps_g, pp_ncol)) if pr_raw is not None else None
        K_g = np.zeros((n_ps_g, 1)) if "K_ps" in pm else None
        C_g = np.zeros((n_ps_g, 1)) if "C_ps" in pm else None
        coh_g = np.zeros((n_ps_g, 1)) if "coh_ps" in pm else None

        K_raw = np.asarray(pm["K_ps"]).ravel() if "K_ps" in pm else None
        C_raw = np.asarray(pm["C_ps"]).ravel() if "C_ps" in pm else None
        coh_raw = np.asarray(pm["coh_ps"]).ravel() if "coh_ps" in pm else None

        for gi in range(n_ps_g):
            sl = slice(f_ix[gi], l_ix[gi] + 1)
            w_snr = ps_snr[sl]
            w_ph = ps_weight[sl]
            ws_ph = w_ph.sum()
            pp_g[gi] = np.sum(pp_data[ix_idx[sl]] * w_snr[:, None], axis=0)
            if pr_g is not None:
                pr_g[gi] = np.sum(pr_raw[ix_idx[sl]] * w_snr[:, None], axis=0)
            if K_g is not None:
                K_g[gi, 0] = np.sum(K_raw[ix_idx[sl]] * w_ph) / ws_ph
            if C_g is not None:
                C_g[gi, 0] = np.sum(C_raw[ix_idx[sl]] * w_ph) / ws_ph
            if coh_g is not None:
                snr_sum = np.sqrt(np.sum(w_snr ** 2))
                if snr_sum > 0 and np.isfinite(snr_sum):
                    coh_g[gi, 0] = np.sqrt(1.0 / (1.0 + 1.0 / snr_sum))
                else:
                    coh_g[gi, 0] = 0.0

        all_ph_patch.append(pp_g.astype(np.complex64))
        if pr_g is not None:
            all_ph_res.append(pr_g.astype(np.float32))
        if K_g is not None:
            all_K_ps.append(K_g)
        if C_g is not None:
            all_C_ps.append(C_g)
        if coh_g is not None:
            all_coh_ps.append(coh_g)

        # bp
        bp = _load_h5_or_mat(pdir, f"bp{v}", ["bperp_mat"])
        bp_data = np.asarray(bp["bperp_mat"], dtype=np.float32)
        bp_g = np.zeros((n_ps_g, bp_data.shape[1]), dtype=np.float64)
        for gi in range(n_ps_g):
            sl = slice(f_ix[gi], l_ix[gi] + 1)
            w = ps_weight[sl].copy()
            w[w == 0] = 1e-9
            bp_g[gi] = np.sum(bp_data[ix_idx[sl]] * w[:, None], axis=0) / w.sum()
        all_bperp_mat.append(bp_g.astype(np.float32))

        # la, inc, hgt
        for arr_list, basename, key in [
            (all_la, f"la{v}", "la"),
            (all_inc, f"inc{v}", "inc"),
            (all_hgt, f"hgt{v}", "hgt"),
        ]:
            arr = _load_optional(pdir, basename, key)
            if arr is not None:
                arr = arr.ravel()
                if arr.shape[0] <= int(ix_idx.max()):
                    logger.warning(
                        "Skipping %s for %s: optional rows %d do not cover selected index %d",
                        basename,
                        pdir.name,
                        arr.shape[0],
                        int(ix_idx.max()),
                    )
                    continue
                arr_g = np.zeros((n_ps_g, 1))
                for gi in range(n_ps_g):
                    sl = slice(f_ix[gi], l_ix[gi] + 1)
                    w = ps_weight[sl]
                    ws = w.sum()
                    if ws > 0 and np.isfinite(ws):
                        arr_g[gi, 0] = np.sum(arr[ix_idx[sl]] * w) / ws
                arr_list.append(arr_g)


# ============================================================================
# Utility: row intersection
# ============================================================================

def _intersect_rows(A: np.ndarray, B: np.ndarray):
    """Find common rows between A and B.

    Returns (common_rows, indices_in_A, indices_in_B),
    similar to MATLAB ``intersect(A, B, 'rows')``.
    """
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty((0, A.shape[1])), np.empty(0, dtype=int), np.empty(0, dtype=int)

    # Use structured arrays for row-wise comparison
    dtype = np.dtype([('f' + str(i), A.dtype) for i in range(A.shape[1])])
    A_struct = np.ascontiguousarray(A).view(dtype).ravel()
    B_struct = np.ascontiguousarray(B.astype(A.dtype)).view(dtype).ravel()

    common, ia, ib = np.intersect1d(A_struct, B_struct, return_indices=True)
    return A[ia], ia, ib


# ============================================================================
# Module-level convenience
# ============================================================================

def ps_merge_patches(project_dir: Union[str, Path], psver: int = 2) -> int:
    """Module-level entry point (mirrors MATLAB ``ps_merge_patches``)."""
    return PatchMerger(project_dir=project_dir, psver=psver).run()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 5b: Merge patches and compute IFG std",
    )
    parser.add_argument("project_dir", type=str, help="Path to project directory")
    parser.add_argument("--psver", type=int, default=2)
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge, only compute IFG std")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if not args.skip_merge:
        ps_merge_patches(args.project_dir, psver=args.psver)
    calc_ifg_std(Path(args.project_dir), psver=args.psver)
