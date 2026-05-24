#!/usr/bin/env python3
"""
scla_estimation.py — StaMPS Step 7: Estimate SCLA & master APS (Python port)
=============================================================================

Estimates the Spatially-Correlated Look Angle (SCLA) error (DEM error)
and optionally co-estimates mean velocity.  Then smooths the result by
clipping outlier K/C values to their Delaunay-neighbor range.

MATLAB correspondence
---------------------
+---------------------------------------+------------------------------------------+
| MATLAB                                | Python                                   |
+=======================================+==========================================+
| ps_calc_scla(use_sb, coest_v)         | SclaEstimator(patch_dir, psver).calc()   |
| ps_smooth_scla(use_sb)                | SclaEstimator.smooth(use_sb)             |
| ps_deramp(ps, ph_all, degree)         | _ps_deramp(xy, ph_all, degree)           |
| ps_setref(ps)                         | _ps_setref(lonlat, xy, cfg)              |
| lscov(G, d, V)  (MATLAB built-in)    | _lscov(G, d, V) via np.linalg            |
+---------------------------------------+------------------------------------------+

Global/appdata mapping:
- MATLAB ``getparm('scla_deramp')``   → ``cfg.getparm('scla_deramp')``
- MATLAB ``getparm('scla_method')``   → ``cfg.getparm('scla_method')``
- MATLAB ``getparm('drop_ifg_index')`` → ``cfg.getparm('drop_ifg_index')``
- MATLAB ``load(psname) → ps``        → ``_load_ps_h5()`` from ps{v}.h5
- MATLAB ``load(phuwname) → uw``      → ``_load_phuw_h5()`` from phuw{v}.h5
- MATLAB ``load(bpname) → bp``        → ``_load_bp_h5()`` from bp{v}.h5

Numerical types:
- MATLAB default double  → Python float64 for design matrices & inversion
- MATLAB single for I/O  → Python float32 for HDF5 storage
- MATLAB int16 for costs  → not used here (that was Step 6)

Refactored from: ps_calc_scla.m, ps_smooth_scla.m, ps_deramp.m, ps_setref.m
  (Andy Hooper, 2006+; David Bekaert, 2013+)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig  # noqa: E402

logger = logging.getLogger("stamps")


# ===========================================================================
# Helper: load HDF5 datasets
# ===========================================================================

def _load_h5(path: Path, keys: List[str]) -> Dict[str, np.ndarray]:
    """Load *keys* from an HDF5 file, returning a dict.  Missing keys → skip."""
    data: Dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as hf:
        for k in keys:
            if k in hf:
                data[k] = np.asarray(hf[k][:])
    return data


def _load_h5_or_mat(directory: Path, basename: str,
                     keys: List[str]) -> Dict[str, np.ndarray]:
    """Try .h5, then fall back to .mat (scipy.io)."""
    h5_path = directory / f"{basename}.h5"
    if h5_path.is_file():
        return _load_h5(h5_path, keys)
    mat_path = directory / f"{basename}.mat"
    if mat_path.is_file():
        from scipy.io import loadmat
        raw = loadmat(str(mat_path), squeeze_me=True)
        return {k: np.asarray(raw[k]) for k in keys if k in raw}
    raise FileNotFoundError(
        f"Neither {h5_path.name} nor {mat_path.name} found in {directory}"
    )


# ===========================================================================
# Helper: parse drop_ifg_index from config (may be string, array, or empty)
# ===========================================================================

def _parse_drop_index(raw) -> np.ndarray:
    """Return a 0-based int array of IFG indices to drop.

    MATLAB stores drop_ifg_index as a double vector (1-based).  In the
    Python config it may arrive as:
      - numpy array of floats/ints (ideal)
      - empty string / bytes / None
      - scalar
    We convert to 0-based int32.
    """
    if raw is None:
        return np.array([], dtype=np.int32)
    if isinstance(raw, (bytes, str)):
        s = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        s = s.strip().strip("[]")
        if not s or all(c == "\x00" for c in s):
            return np.array([], dtype=np.int32)
        try:
            vals = [int(float(x)) for x in s.replace(",", " ").split()]
        except ValueError:
            return np.array([], dtype=np.int32)
        arr = np.array(vals, dtype=np.int32)
    else:
        arr = np.atleast_1d(np.asarray(raw).ravel())
        if arr.dtype.kind in ("U", "S"):
            return np.array([], dtype=np.int32)
        arr = arr.astype(np.int32)
    if arr.size == 0 or np.all(arr == 0):
        return np.array([], dtype=np.int32)
    # MATLAB is 1-based; convert to 0-based
    if arr.size > 0 and arr.min() >= 1:
        arr = arr - 1
    return arr[arr >= 0]


# ===========================================================================
# Helper: reference PS selection  (MATLAB ps_setref.m)
# ===========================================================================

def _ps_setref(lonlat: np.ndarray, xy: np.ndarray,
               cfg: StampsConfig) -> np.ndarray:
    """Return indices of reference PS pixels.

    MATLAB logic (ps_setref.m):
      1. If ref_x / ref_y exist → box filter on xy(:,2:3)
      2. Else → box filter on lonlat using ref_lon / ref_lat
         with optional circular radius from ref_centre_lonlat / ref_radius
      3. If nothing selected → all PS are reference (warn)
    """
    # Try ref_x / ref_y first
    ref_x = cfg.getparm("ref_x")
    ref_y = cfg.getparm("ref_y")
    if ref_x is not None and ref_y is not None:
        ref_x = np.atleast_1d(np.asarray(ref_x, dtype=np.float64)).ravel()
        ref_y = np.atleast_1d(np.asarray(ref_y, dtype=np.float64)).ravel()
        if ref_x.size >= 2 and ref_y.size >= 2:
            mask = ((xy[:, 1] > ref_x[0]) & (xy[:, 1] < ref_x[1]) &
                    (xy[:, 2] > ref_y[0]) & (xy[:, 2] < ref_y[1]))
            ref_ps = np.where(mask)[0]
            if ref_ps.size > 0:
                logger.info("  %d ref PS selected (ref_x/ref_y)", ref_ps.size)
                return ref_ps

    # Lon/lat box
    ref_lon = cfg.getparm("ref_lon")
    ref_lat = cfg.getparm("ref_lat")
    if ref_lon is None:
        ref_lon = np.array([-np.inf, np.inf])
    else:
        ref_lon = np.atleast_1d(np.asarray(ref_lon, dtype=np.float64)).ravel()
    if ref_lat is None:
        ref_lat = np.array([-np.inf, np.inf])
    else:
        ref_lat = np.atleast_1d(np.asarray(ref_lat, dtype=np.float64)).ravel()

    ref_radius = cfg.getparm("ref_radius")
    if ref_radius is not None and float(ref_radius) == float("-inf"):
        logger.info("  No reference set (ref_radius = -inf)")
        return np.arange(lonlat.shape[0])

    mask = ((lonlat[:, 0] > ref_lon[0]) & (lonlat[:, 0] < ref_lon[1]) &
            (lonlat[:, 1] > ref_lat[0]) & (lonlat[:, 1] < ref_lat[1]))
    ref_ps = np.where(mask)[0]

    if ref_radius is not None and np.isfinite(float(ref_radius)):
        ref_centre = cfg.getparm("ref_centre_lonlat")
        if ref_centre is not None:
            ref_centre = np.atleast_1d(np.asarray(ref_centre, dtype=np.float64)).ravel()
            # Approximate lon/lat → local meters (simplified, not full llh2local)
            ll0 = np.mean(lonlat, axis=0)
            cos_lat = np.cos(np.radians(ll0[1]))
            dx = (lonlat[ref_ps, 0] - ref_centre[0]) * cos_lat * 111319.5
            dy = (lonlat[ref_ps, 1] - ref_centre[1]) * 111319.5
            dist_sq = dx ** 2 + dy ** 2
            ref_ps = ref_ps[dist_sq <= float(ref_radius) ** 2]

    if ref_ps.size == 0:
        logger.warning("  No ref PS found; using ALL %d PS as reference", lonlat.shape[0])
        ref_ps = np.arange(lonlat.shape[0])
    else:
        logger.info("  %d ref PS selected", ref_ps.size)
    return ref_ps


# ===========================================================================
# Helper: deramping  (MATLAB ps_deramp.m)
# ===========================================================================

def _ps_deramp(xy: np.ndarray, ph_all: np.ndarray,
               degree: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Remove polynomial ramp from each IFG.

    MATLAB ps_deramp.m:
      degree 1:   z = ax + by + c
      degree 1.5: z = ax + by + cxy + d
      degree 2:   z = ax² + by² + cxy + d
      degree 3:   z = ax³ + by³ + cx²y + dy²x + ex² + fy² + gxy + h

    Parameters
    ----------
    xy : (n_ps, 3) — column 0 = id, columns 1,2 = x, y
    ph_all : (n_ps, n_ifg) — phase to deramp (modified in place)
    degree : ramp polynomial degree

    Returns
    -------
    ph_deramped, ph_ramp  — both (n_ps, n_ifg)
    """
    n_ps = xy.shape[0]
    xk = xy[:, 1].astype(np.float64) / 1000.0
    yk = xy[:, 2].astype(np.float64) / 1000.0

    if degree == 1:
        A = np.column_stack([xk, yk, np.ones(n_ps)])
        logger.info("  Deramp: z = ax + by + c")
    elif degree == 1.5:
        A = np.column_stack([xk, yk, xk * yk, np.ones(n_ps)])
        logger.info("  Deramp: z = ax + by + cxy + d")
    elif degree == 2:
        A = np.column_stack([xk ** 2, yk ** 2, xk * yk, np.ones(n_ps)])
        logger.info("  Deramp: z = ax^2 + by^2 + cxy + d")
    elif degree == 3:
        A = np.column_stack([
            xk ** 3, yk ** 3, xk ** 2 * yk, yk ** 2 * xk,
            xk ** 2, yk ** 2, xk * yk, np.ones(n_ps)
        ])
        logger.info("  Deramp: z = ax^3 + by^3 + ... + h")
    else:
        A = np.column_stack([xk, yk, np.ones(n_ps)])
        logger.info("  Deramp: z = ax + by + c (fallback)")

    ph_ramp = np.full_like(ph_all, np.nan, dtype=np.float64)
    ph_deramped = ph_all.astype(np.float64).copy()

    for k in range(ph_all.shape[1]):
        col = ph_deramped[:, k]
        valid = np.isfinite(col)
        if valid.sum() > A.shape[1] + 2:  # need enough points
            coeff, _, _, _ = np.linalg.lstsq(A[valid], col[valid], rcond=None)
            ramp = A @ coeff
            ph_ramp[:, k] = ramp
            ph_deramped[:, k] = col - ramp
        else:
            logger.warning("  IFG %d not deramped (too few valid points)", k + 1)

    return ph_deramped, ph_ramp


# ===========================================================================
# Helper: weighted least squares  (MATLAB lscov equivalent)
# ===========================================================================

def _lscov(G: np.ndarray, d: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Weighted least squares: minimise (d - G m)' V^{-1} (d - G m).

    MATLAB ``lscov(G, d, V)`` uses the covariance matrix V.
    Equivalent to solving the normal equations:
        G' V^{-1} G m = G' V^{-1} d

    For large n_ps we vectorise: d is (n_obs, n_ps).

    Parameters
    ----------
    G : (n_obs, n_param)
    d : (n_obs, n_ps) — each column is one pixel's observation vector
    V : (n_obs, n_obs) — covariance matrix (or identity)

    Returns
    -------
    m : (n_param, n_ps) — parameter estimates per pixel
    """
    # Check if V is identity → simple lstsq (much faster)
    if np.allclose(V, np.eye(V.shape[0])):
        m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
        return m

    # Weighted: W = V^{-1}
    try:
        L = np.linalg.cholesky(V)
        # Solve L x = G and L x = d for whitened system
        Gw = np.linalg.solve(L, G)
        dw = np.linalg.solve(L, d)
        m, _, _, _ = np.linalg.lstsq(Gw, dw, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback: direct normal equations
        Vinv = np.linalg.pinv(V)
        GtVinv = G.T @ Vinv
        m = np.linalg.solve(GtVinv @ G, GtVinv @ d)
    return m


# ===========================================================================
# Main class: SclaEstimator
# ===========================================================================

class SclaEstimator:
    """Step 7: Spatially-Correlated Look Angle error estimation.

    Mirrors the MATLAB sequence in stamps.m (lines 509-526):
      Non-SB:  ps_calc_scla(0, 1)  →  ps_smooth_scla()
      SB:      ps_calc_scla(1, 1)  →  ps_smooth_scla(1)  →  ps_calc_scla(0, 1)
    """

    def __init__(self, patch_dir: Path, psver: int = 2):
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver
        self.cfg = StampsConfig(work_dir=self.patch_dir)
        if not self.cfg._loaded:
            self.cfg.load()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full Step 7 pipeline."""
        sb_flag = str(self.cfg.getparm("small_baseline_flag") or "n").strip().lower()

        if sb_flag == "y":
            logger.info("SB mode: ps_calc_scla(1,1) → ps_smooth_scla(1) → ps_calc_scla(0,1)")
            self.calc_scla(use_small_baselines=True, coest_mean_vel=True)
            self.smooth_scla(use_small_baselines=True)
            self.calc_scla(use_small_baselines=False, coest_mean_vel=True)
        else:
            logger.info("SM mode: ps_calc_scla(0,1) → ps_smooth_scla()")
            self.calc_scla(use_small_baselines=False, coest_mean_vel=True)
            self.smooth_scla(use_small_baselines=False)

    # ------------------------------------------------------------------
    # ps_calc_scla  (MATLAB ps_calc_scla.m)
    # ------------------------------------------------------------------

    def calc_scla(self, use_small_baselines: bool = False,
                  coest_mean_vel: bool = False) -> None:
        """Estimate SCLA coefficient K_ps_uw and master APS C_ps_uw.

        MATLAB correspondence: ``ps_calc_scla(use_small_baselines, coest_mean_vel)``

        Algorithm:
          1. Load ps, bp, phuw data
          2. Optional: subtract tropospheric correction (subtr_tropo)
          3. Optional: deramp interferograms (scla_deramp)
          4. Set reference PS, subtract reference mean
          5. Build bperp_mat with master zero column
          6. Compute sequential phase/bperp differences (SM) or direct (SB)
          7. Build design matrix G = [1, bperp, (day)]
          8. Solve lscov(G, ph, V) for K_ps_uw
          9. Compute ph_scla, C_ps_uw
          10. Save to scla{psver}.h5 or scla_sb{psver}.h5
        """
        logger.info("Estimating spatially-correlated look angle error...")
        v = self.psver
        cfg = self.cfg

        # ---- Configuration parameters ----
        sb_flag = str(cfg.getparm("small_baseline_flag") or "n").strip().lower()
        drop_ifg_index = _parse_drop_index(cfg.getparm("drop_ifg_index"))
        scla_method = str(cfg.getparm("scla_method") or "L2").strip().upper()
        scla_deramp = str(cfg.getparm("scla_deramp") or "n").strip().lower()
        subtr_tropo = str(cfg.getparm("subtr_tropo") or "n").strip().lower()

        if use_small_baselines:
            scla_drop = _parse_drop_index(cfg.getparm("sb_scla_drop_index"))
            logger.info("  Using small baseline interferograms")
        else:
            scla_drop = _parse_drop_index(cfg.getparm("scla_drop_index"))

        # ---- Load data ----
        ps = self._load_ps(v)
        bp = self._load_bp(v, ps)
        uw = self._load_phuw(v, use_small_baselines)

        n_ps = ps["n_ps"]
        n_ifg = ps["n_ifg"]
        n_image = ps["n_image"]
        master_ix = ps["master_ix"]  # 1-based in MATLAB, convert below

        # MATLAB master_ix is 1-based; Python uses 0-based
        master_ix_0 = master_ix - 1

        # ph_uw shape: (n_ps, n_ifg) or (n_ps, n_image)
        ph_uw = uw["ph_uw"].astype(np.float64)

        # ---- Build unwrap_ifg_index (0-based) ----
        if sb_flag == "y" and not use_small_baselines:
            # SM from SB: use all n_image columns
            unwrap_ifg_index = np.arange(n_image)
            n_ifg_work = n_image
        else:
            all_ifg = np.arange(n_ifg)
            unwrap_ifg_index = np.setdiff1d(all_ifg, drop_ifg_index)
            n_ifg_work = n_ifg

        # ---- Optional: subtract tropospheric correction ----
        if subtr_tropo == "y":
            ph_uw = self._subtract_tropo(ph_uw, v, use_small_baselines)

        # ---- Optional: deramp ----
        ph_ramp: Optional[np.ndarray] = None
        if scla_deramp == "y":
            logger.info("  Deramping interferograms...")
            deramp_degree = self._get_deramp_degree()
            _, ph_ramp = _ps_deramp(ps["xy"], ph_uw, degree=deramp_degree)
            ph_uw = ph_uw - np.nan_to_num(ph_ramp, nan=0.0)

        # ---- Remove scla_drop_index ----
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop)

        # ---- Set reference PS & subtract reference mean ----
        ref_ps = _ps_setref(ps["lonlat"], ps["xy"], cfg)
        if ref_ps.size > 0 and ref_ps.max() < n_ps:
            ref_mean = np.nanmean(ph_uw[ref_ps, :], axis=0, keepdims=True)
            ph_uw = ph_uw - ref_mean

        # ---- Build bperp_mat and phase/bperp for inversion ----
        bperp_mat = bp["bperp_mat"].astype(np.float64)

        if not use_small_baselines:
            # Single-master mode
            if sb_flag == "y":
                # SM from SB inversion: reconstruct SM bperp from SB bperp via lstsq
                # Build SB→SM design matrix G  (MATLAB ps_calc_scla.m lines 165-176)
                bperp_mat_sm = np.zeros((n_ps, n_image), dtype=np.float64)
                if "ifgday_ix" in ps and ps["ifgday_ix"] is not None:
                    ifgday_ix = ps["ifgday_ix"]  # (n_ifg, 2) 0-based
                    G_sb = np.zeros((ps["n_ifg"], n_image), dtype=np.float64)
                    for i in range(ps["n_ifg"]):
                        G_sb[i, ifgday_ix[i, 0]] = -1
                        G_sb[i, ifgday_ix[i, 1]] = 1

                    # Remove master_ix from unwrap_ifg_index (MATLAB line 173)
                    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, np.array([master_ix_0]))
                    G_sub = G_sb[:, unwrap_ifg_index]
                    # Solve: G_sub \ bperp_mat' → bperp for each PS
                    bperp_some, _, _, _ = np.linalg.lstsq(
                        G_sub, bperp_mat.T, rcond=None
                    )
                    bperp_mat_sm[:, unwrap_ifg_index] = bperp_some.T
                bperp_mat = bperp_mat_sm
            else:
                # Regular PS: insert zero column for master
                # bp.bperp_mat is (n_ps, n_ifg-1) → insert zero at master_ix_0
                if bperp_mat.shape[1] == n_ifg - 1:
                    bperp_mat = np.insert(bperp_mat, master_ix_0, 0.0, axis=1)
                elif bperp_mat.shape[1] == n_ifg:
                    # Already has master column (may be non-zero)
                    bperp_mat[:, master_ix_0] = 0.0

            # Sequential differences to reduce influence of deformation
            # MATLAB: day = diff(ps.day(unwrap_ifg_index));
            #         ph  = diff(uw.ph_uw(:, unwrap_ifg_index), [], 2);
            #         bperp = diff(bperp_mat(:, unwrap_ifg_index), [], 2);
            day_vec = ps["day"]  # (n_image,) or (n_ifg,)
            day_diff = np.diff(day_vec[unwrap_ifg_index]).astype(np.float64)
            ph = np.diff(ph_uw[:, unwrap_ifg_index], axis=1)
            bperp_diff = np.diff(bperp_mat[:, unwrap_ifg_index], axis=1)
        else:
            # Small-baseline mode: use IFGs directly
            bperp_diff = bperp_mat[:, unwrap_ifg_index]
            # day = ifgday(unwrap_ifg_index, 2) - ifgday(unwrap_ifg_index, 1)
            if "ifgday" in ps and ps["ifgday"] is not None:
                day_diff = (ps["ifgday"][unwrap_ifg_index, 1] -
                            ps["ifgday"][unwrap_ifg_index, 0]).astype(np.float64)
            else:
                day_diff = np.ones(unwrap_ifg_index.size, dtype=np.float64)
            ph = ph_uw[:, unwrap_ifg_index]

        bperp_mean = np.mean(bperp_diff, axis=0)  # (n_seq,)
        n_seq = ph.shape[1]
        logger.info("  %d IFGs used in estimation", n_seq)

        # ---- Build design matrix G ----
        # MATLAB: G = [ones, mean_bperp', (day)]
        if not coest_mean_vel or n_seq < 4:
            G = np.column_stack([np.ones(n_seq), bperp_mean])
        else:
            G = np.column_stack([np.ones(n_seq), bperp_mean, day_diff])

        # ---- Build ifg covariance matrix ----
        ifg_vcm = np.eye(n_ifg_work, dtype=np.float64)
        if sb_flag == "y":
            # Try loading sm_cov or sb_cov from phuw_sb_res
            ifg_vcm = self._load_ifg_vcm_sb(v, use_small_baselines, n_ifg_work)
        else:
            # Try ifgstd file
            ifg_vcm = self._load_ifg_vcm_std(v, n_ifg_work)

        # For SM mode, MATLAB uses identity for the sequential differences
        if not use_small_baselines:
            ifg_vcm_use = np.eye(n_seq, dtype=np.float64)
        else:
            # Sub-select the unwrap_ifg_index rows/cols
            idx = unwrap_ifg_index[unwrap_ifg_index < ifg_vcm.shape[0]]
            ifg_vcm_use = ifg_vcm[np.ix_(idx, idx)] if idx.size > 0 else np.eye(n_seq)

        # ---- Solve: m = lscov(G, ph', ifg_vcm_use) ----
        # ph shape: (n_ps, n_seq);  lscov wants (n_seq, n_ps)
        m = _lscov(G, ph.T, ifg_vcm_use)  # (n_param, n_ps)

        K_ps_uw = m[1, :].astype(np.float64)  # SCLA coefficient per PS

        # ---- Optional L1 refinement ----
        if scla_method == "L1":
            logger.info("  Refining with L1 norm (fmin)...")
            from scipy.optimize import minimize
            for i in range(n_ps):
                d = ph[i, :]
                m0 = m[:, i].copy()
                res = minimize(lambda x: np.sum(np.abs(d - G @ x)), m0,
                               method="Nelder-Mead",
                               options={"maxiter": 1000, "xatol": 1e-6})
                K_ps_uw[i] = res.x[1]
                if (i + 1) % 10000 == 0:
                    logger.info("    %d of %d pixels processed", i + 1, n_ps)

        # ---- Compute ph_scla ----
        # MATLAB: ph_scla = K_ps_uw * bperp_mat  (element-wise broadcast)
        ph_scla = K_ps_uw[:, np.newaxis] * bperp_mat  # (n_ps, n_image or n_ifg)

        # ---- Compute C_ps_uw (master APS + orbit error) ----
        if not use_small_baselines:
            # Remove master index from unwrap list for residual computation
            uw_idx_no_master = np.setdiff1d(unwrap_ifg_index, np.array([master_ix_0]))
            residual = ph_uw[:, uw_idx_no_master] - ph_scla[:, uw_idx_no_master]
            if not coest_mean_vel:
                C_ps_uw = np.nanmean(residual, axis=1)
            else:
                # Co-estimate velocity: solve [1, day] for residual
                day_all = ps["day"].astype(np.float64)
                t = day_all[uw_idx_no_master] - day_all[master_ix_0]
                G_c = np.column_stack([np.ones(t.size), t])
                idx_c = uw_idx_no_master
                if idx_c.max() < ifg_vcm.shape[0]:
                    vcm_c = ifg_vcm[np.ix_(idx_c, idx_c)]
                else:
                    vcm_c = np.eye(idx_c.size)
                m_c = _lscov(G_c, residual.T, vcm_c)
                C_ps_uw = m_c[0, :]
        else:
            C_ps_uw = np.zeros(n_ps, dtype=np.float64)

        # ---- Save results ----
        suffix = "_sb" if use_small_baselines else ""
        scla_name = f"scla{suffix}{v}"
        save_path = self.patch_dir / f"{scla_name}.h5"
        with h5py.File(str(save_path), "w") as hf:
            hf.create_dataset("ph_scla", data=ph_scla.astype(np.float32),
                              compression="gzip")
            hf.create_dataset("K_ps_uw", data=K_ps_uw.astype(np.float32))
            hf.create_dataset("C_ps_uw", data=C_ps_uw.astype(np.float32))
            if ph_ramp is not None:
                hf.create_dataset("ph_ramp", data=ph_ramp.astype(np.float32),
                                  compression="gzip")
            hf.create_dataset("ifg_vcm", data=ifg_vcm.astype(np.float32),
                              compression="gzip")
        logger.info("  Saved %s (K_ps_uw %s, C_ps_uw %s, ph_scla %s)",
                    save_path.name, K_ps_uw.shape, C_ps_uw.shape, ph_scla.shape)

    # ------------------------------------------------------------------
    # ps_smooth_scla  (MATLAB ps_smooth_scla.m)
    # ------------------------------------------------------------------

    def smooth_scla(self, use_small_baselines: bool = False) -> None:
        """Smooth SCLA by clipping outliers to Delaunay-neighbor range.

        MATLAB correspondence: ``ps_smooth_scla(use_small_baselines)``

        Algorithm:
          1. Build Delaunay triangulation from PS xy coordinates
          2. Extract unique edges
          3. For each edge, track min/max neighbor K_ps_uw and C_ps_uw
          4. Clip outliers to neighbor bounds
          5. Recompute ph_scla from smoothed K_ps_uw
          6. Save to scla_smooth{v}.h5 or scla_smooth_sb{v}.h5
        """
        logger.info("Smoothing spatially-correlated look angle error...")
        v = self.psver
        sb_flag = str(self.cfg.getparm("small_baseline_flag") or "n").strip().lower()

        # ---- Load data ----
        ps = self._load_ps(v)
        suffix = "_sb" if use_small_baselines else ""
        scla_data = _load_h5(self.patch_dir / f"scla{suffix}{v}.h5",
                             ["K_ps_uw", "C_ps_uw", "ph_ramp"])

        K_ps_uw = scla_data["K_ps_uw"].astype(np.float64).ravel()
        C_ps_uw = scla_data["C_ps_uw"].astype(np.float64).ravel()
        ph_ramp = scla_data.get("ph_ramp", None)

        n_ps = ps["n_ps"]
        xy = ps["xy"].astype(np.float64)

        # ---- Build Delaunay triangulation → edges ----
        # MATLAB uses triangle/delaunay → extract unique edges
        logger.info("  Number of points per ifg: %d", n_ps)
        tri = Delaunay(xy[:, 1:3])  # xy columns 1,2 = x,y

        # Extract unique edges from simplices
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges_set.add((min(a, b), max(a, b)))
        edges = np.array(list(edges_set), dtype=np.int32)
        n_edge = edges.shape[0]
        logger.info("  Number of arcs per ifg: %d", n_edge)

        # ---- Find min/max neighbor for each PS ----
        # MATLAB: for each edge, update Kneigh_min/max and Cneigh_min/max
        Kneigh_min = np.full(n_ps, np.inf, dtype=np.float64)
        Kneigh_max = np.full(n_ps, -np.inf, dtype=np.float64)
        Cneigh_min = np.full(n_ps, np.inf, dtype=np.float64)
        Cneigh_max = np.full(n_ps, -np.inf, dtype=np.float64)

        # Vectorized neighbor min/max computation
        e0, e1 = edges[:, 0], edges[:, 1]
        # For edge (a, b): node a gets neighbor value from b, node b from a
        # We use np.minimum.at / np.maximum.at for scatter-reduce
        np.minimum.at(Kneigh_min, e0, K_ps_uw[e1])
        np.minimum.at(Kneigh_min, e1, K_ps_uw[e0])
        np.maximum.at(Kneigh_max, e0, K_ps_uw[e1])
        np.maximum.at(Kneigh_max, e1, K_ps_uw[e0])
        np.minimum.at(Cneigh_min, e0, C_ps_uw[e1])
        np.minimum.at(Cneigh_min, e1, C_ps_uw[e0])
        np.maximum.at(Cneigh_max, e0, C_ps_uw[e1])
        np.maximum.at(Cneigh_max, e1, C_ps_uw[e0])

        # ---- Clip outliers ----
        K_clip_hi = K_ps_uw > Kneigh_max
        K_clip_lo = K_ps_uw < Kneigh_min
        K_ps_uw[K_clip_hi] = Kneigh_max[K_clip_hi]
        K_ps_uw[K_clip_lo] = Kneigh_min[K_clip_lo]
        logger.info("  K_ps_uw: %d clipped high, %d clipped low",
                    K_clip_hi.sum(), K_clip_lo.sum())

        C_clip_hi = C_ps_uw > Cneigh_max
        C_clip_lo = C_ps_uw < Cneigh_min
        C_ps_uw[C_clip_hi] = Cneigh_max[C_clip_hi]
        C_ps_uw[C_clip_lo] = Cneigh_min[C_clip_lo]
        logger.info("  C_ps_uw: %d clipped high, %d clipped low",
                    C_clip_hi.sum(), C_clip_lo.sum())

        # ---- Recompute ph_scla from smoothed K ----
        bp = self._load_bp(v, ps)
        bperp_mat = bp["bperp_mat"].astype(np.float64)

        if not use_small_baselines:
            master_ix_0 = ps["master_ix"] - 1
            n_image = ps["n_image"]
            if sb_flag == "y":
                # SM from SB: reconstruct SM bperp
                bperp_mat_sm = np.zeros((n_ps, n_image), dtype=np.float64)
                if "ifgday_ix" in ps and ps["ifgday_ix"] is not None:
                    ifgday_ix = ps["ifgday_ix"]
                    n_ifg_sb = ps["n_ifg"]
                    G_sb = np.zeros((n_ifg_sb, n_image), dtype=np.float64)
                    for i in range(n_ifg_sb):
                        G_sb[i, ifgday_ix[i, 0]] = -1
                        G_sb[i, ifgday_ix[i, 1]] = 1
                    G_sb_sub = G_sb[:, np.setdiff1d(np.arange(n_image),
                                                    np.array([master_ix_0]))]
                    bperp_some, _, _, _ = np.linalg.lstsq(
                        G_sb_sub, bperp_mat.T, rcond=None
                    )
                    idx_no_master = np.setdiff1d(np.arange(n_image),
                                                 np.array([master_ix_0]))
                    bperp_mat_sm[:, idx_no_master] = bperp_some.T
                    # Insert zero master column
                    bperp_mat_sm[:, master_ix_0] = 0.0
                bperp_mat = bperp_mat_sm
            else:
                # Regular PS: insert zero column for master if needed
                if bperp_mat.shape[1] == ps["n_ifg"] - 1:
                    bperp_mat = np.insert(bperp_mat, master_ix_0, 0.0, axis=1)
                elif bperp_mat.shape[1] == ps["n_ifg"]:
                    bperp_mat[:, master_ix_0] = 0.0

        ph_scla = K_ps_uw[:, np.newaxis] * bperp_mat

        # ---- Save ----
        smooth_name = f"scla_smooth{suffix}{v}"
        save_path = self.patch_dir / f"{smooth_name}.h5"
        with h5py.File(str(save_path), "w") as hf:
            hf.create_dataset("K_ps_uw", data=K_ps_uw.astype(np.float32))
            hf.create_dataset("C_ps_uw", data=C_ps_uw.astype(np.float32))
            hf.create_dataset("ph_scla", data=ph_scla.astype(np.float32),
                              compression="gzip")
            if ph_ramp is not None:
                hf.create_dataset("ph_ramp", data=ph_ramp.astype(np.float32),
                                  compression="gzip")
        logger.info("  Saved %s (K_ps_uw %s, ph_scla %s)",
                    save_path.name, K_ps_uw.shape, ph_scla.shape)

    # ------------------------------------------------------------------
    # Private data loaders
    # ------------------------------------------------------------------

    def _load_ps(self, v: int) -> Dict[str, Any]:
        """Load PS metadata from ps{v}.h5 (or .mat fallback).

        Returns dict with: n_ps, n_ifg, n_image, master_ix, xy, lonlat,
                           day, bperp, ifgday_ix (optional), ifgday (optional).
        """
        keys = ["n_ps", "n_ifg", "n_image", "master_ix", "xy", "lonlat",
                "day", "bperp", "ifgday_ix", "ifgday", "day_ix"]
        raw = _load_h5_or_mat(self.patch_dir, f"ps{v}", keys)

        # Scalar extraction
        for scalar_key in ["n_ps", "n_ifg", "n_image", "master_ix"]:
            if scalar_key in raw:
                raw[scalar_key] = int(np.asarray(raw[scalar_key]).ravel()[0])

        # Ensure 1D for day, bperp
        for k in ["day", "bperp"]:
            if k in raw:
                raw[k] = np.asarray(raw[k]).ravel()

        # Ensure 2D for xy, lonlat
        for k in ["xy", "lonlat"]:
            if k in raw:
                arr = np.asarray(raw[k])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 2) if k == "lonlat" else arr.reshape(-1, 3)
                raw[k] = arr

        # ifgday_ix: ensure 2D (n_ifg, 2) and 0-based
        if "ifgday_ix" in raw:
            iix = np.asarray(raw["ifgday_ix"])
            if iix.ndim == 1 and iix.size % 2 == 0:
                iix = iix.reshape(-1, 2)
            elif iix.ndim == 2 and iix.shape[1] == 1:
                # day_ix style: just sequential indices
                iix = None
            raw["ifgday_ix"] = iix

        return raw

    def _load_bp(self, v: int, ps: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Load bperp_mat from bp{v}.h5 (or .mat, or derive from ps.bperp)."""
        try:
            bp = _load_h5_or_mat(self.patch_dir, f"bp{v}", ["bperp_mat"])
            if "bperp_mat" in bp:
                return bp
        except FileNotFoundError:
            pass

        # Fallback: derive from ps.bperp (MATLAB ps_calc_scla.m lines 90-95)
        bperp = ps["bperp"]
        sb_flag = str(self.cfg.getparm("small_baseline_flag") or "n").strip().lower()
        if sb_flag != "y":
            # Remove master entry
            master_ix_0 = ps["master_ix"] - 1
            bperp = np.delete(bperp, master_ix_0)
        bperp_mat = np.tile(bperp, (ps["n_ps"], 1)).astype(np.float32)
        return {"bperp_mat": bperp_mat}

    def _load_phuw(self, v: int,
                   use_small_baselines: bool) -> Dict[str, np.ndarray]:
        """Load unwrapped phase from phuw{v}.h5 or phuw_sb{v}.h5."""
        if use_small_baselines:
            basename = f"phuw_sb{v}"
        else:
            basename = f"phuw{v}"
        return _load_h5_or_mat(self.patch_dir, basename, ["ph_uw", "msd"])

    def _load_ifg_vcm_std(self, v: int, n_ifg: int) -> np.ndarray:
        """Load ifg covariance from ifgstd{v}.h5/.mat (PS mode).

        MATLAB (ps_calc_scla.m lines 228-232):
          ifgstd = load(ifgstdname);
          ifg_vcm = diag((ifgstd.ifg_std * pi/180).^2);
        """
        for d in [self.patch_dir, self.patch_dir.parent]:
            try:
                data = _load_h5_or_mat(d, f"ifgstd{v}", ["ifg_std"])
                if "ifg_std" in data:
                    std = np.asarray(data["ifg_std"]).ravel().astype(np.float64)
                    sigma_sq = (std * np.pi / 180.0) ** 2
                    n = min(sigma_sq.size, n_ifg)
                    vcm = np.eye(n_ifg, dtype=np.float64)
                    vcm[:n, :n] = np.diag(sigma_sq[:n])
                    logger.info("  Loaded ifg_vcm from ifgstd%d (diagonal, %d IFGs)", v, n)
                    return vcm
            except FileNotFoundError:
                continue
        return np.eye(n_ifg, dtype=np.float64)

    def _load_ifg_vcm_sb(self, v: int, use_small_baselines: bool,
                         n_ifg: int) -> np.ndarray:
        """Load SB covariance from phuw_sb_res{v} (sm_cov or sb_cov).

        MATLAB (ps_calc_scla.m lines 215-226):
          phuwres = load(phuwsbresname, 'sm_cov'/'sb_cov');
        """
        cov_key = "sb_cov" if use_small_baselines else "sm_cov"
        try:
            data = _load_h5_or_mat(self.patch_dir, f"phuw_sb_res{v}", [cov_key])
            if cov_key in data:
                cov = np.asarray(data[cov_key], dtype=np.float64)
                logger.info("  Loaded %s from phuw_sb_res%d", cov_key, v)
                return cov
        except FileNotFoundError:
            pass
        return np.eye(n_ifg, dtype=np.float64)

    def _subtract_tropo(self, ph_uw: np.ndarray, v: int,
                        use_small_baselines: bool) -> np.ndarray:
        """Subtract tropospheric correction if available.

        MATLAB (ps_calc_scla.m lines 107-124):
          aps = load(apsname);
          [aps_corr, ...] = ps_plot_tca(aps, tropo_method);
          uw.ph_uw = uw.ph_uw - aps_corr;

        For now, tries to load tca{v}.h5/.mat and subtract ph_tropo_* field.
        Falls back to identity (no correction) if not found.
        """
        suffix = "_sb" if use_small_baselines else ""
        tca_name = f"tca{suffix}{v}"
        try:
            tca = _load_h5_or_mat(self.patch_dir, tca_name,
                                  ["ph_tropo_era", "ph_tropo_era5",
                                   "ph_tropo_gacos", "aps_corr"])
            # Use first available correction
            for key in ["aps_corr", "ph_tropo_era5", "ph_tropo_era",
                        "ph_tropo_gacos"]:
                if key in tca:
                    corr = np.asarray(tca[key], dtype=np.float64)
                    if corr.shape == ph_uw.shape:
                        logger.info("  Subtracted tropospheric correction (%s)", key)
                        return ph_uw - corr
                    else:
                        logger.warning("  Tropo correction shape %s != ph_uw %s; skipping",
                                       corr.shape, ph_uw.shape)
        except FileNotFoundError:
            logger.info("  No tropospheric correction file found; skipping")
        return ph_uw

    def _get_deramp_degree(self) -> float:
        """Get deramping degree (default 1, or from deramp_degree.mat)."""
        dd_path = self.patch_dir / "deramp_degree.mat"
        if dd_path.is_file():
            try:
                from scipy.io import loadmat
                dd = loadmat(str(dd_path), squeeze_me=True)
                if "degree" in dd:
                    deg = float(dd["degree"])
                    logger.info("  Deramp degree from file: %s", deg)
                    return deg
            except Exception:
                pass
        return 1.0


def ps_calc_scla(
    patch_dir: Union[str, Path],
    psver: int = 2,
) -> None:
    """Module-level Step-7 entry point."""
    SclaEstimator(Path(patch_dir), psver=psver).run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 7: Estimate SCLA and smooth"
    )
    parser.add_argument("patch_dir", type=str, help="Project/patch directory")
    parser.add_argument("--psver", type=int, default=2)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    ps_calc_scla(args.patch_dir, psver=args.psver)
