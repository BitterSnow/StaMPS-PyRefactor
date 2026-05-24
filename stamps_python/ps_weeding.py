#!/usr/bin/env python3
"""
ps_weeding.py — Python port of ps_weed.m  (StaMPS Step 4)
===========================================================

Weed out neighbouring and noisy PS pixels, then save a cleaned version
(psver+1) of all data files.

MATLAB → Python mapping
-----------------------
+----------------------------------------+--------------------------------------------------+
| MATLAB concept                         | Python mapping                                   |
+========================================+==================================================+
| ``ps_weed(all_da, no_adj, no_noisy)``  | ``PSWeeder(patch_dir).run(…)``                   |
+----------------------------------------+--------------------------------------------------+
| ``neigh_ix`` 2-D grid + BFS merging    | ``scipy.spatial.cKDTree`` for fast adjacency     |
+----------------------------------------+--------------------------------------------------+
| ``delaunay`` / ``triangle`` edges       | ``scipy.spatial.Delaunay`` + edge extraction     |
+----------------------------------------+--------------------------------------------------+
| ``lscov(G, dph, w)``                   | ``np.linalg.lstsq`` (weighted)                  |
+----------------------------------------+--------------------------------------------------+
| ``stamps_save('weed1', …)``            | ``_save_weed_h5(…, weed1.h5)``                  |
+----------------------------------------+--------------------------------------------------+
| ``save('ps2', '-struct', ps)``         | ``_save_ps2_h5(…)``                             |
+----------------------------------------+--------------------------------------------------+

Algorithm overview (ps_weed.m)
------------------------------
1. **Load** Step-1 (ps1), Step-2 (pm1), Step-3 (select1) data.
2. Extract the *kept* PS subset from Step 3 (``ix2 = sl.ix(sl.keep_ix)``).
3. **Weed adjacent** (optional): remove pixels sharing the same resolution
   cell, keeping the one with highest coherence.
4. **Weed zero elevation** (optional): remove pixels with height < 1e-6.
5. **Remove duplicates**: drop pixels with identical xy coordinates, keeping
   the highest-coherence one.
6. **Weed noisy** (optional): build Delaunay triangulation, compute arc phase
   noise, discard pixels whose minimum-arc std or max noise exceeds thresholds.
7. **Save** cleaned outputs as version 2 files (ps2, pm2, ph2, bp2, hgt2, inc2,
   weed1).

Refactored from: ps_weed.m (Andy Hooper, June 2006)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.spatial import Delaunay

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig, load_mat  # noqa: E402

logger = logging.getLogger("stamps")


def _as_matlab_index_vector(values: np.ndarray) -> np.ndarray:
    """Return an int32 vector using MATLAB 1-based indexing."""
    arr = np.asarray(values).ravel().astype(np.int32)
    if arr.size > 0 and arr.min() == 0:
        arr = arr + 1
    return arr


def _parse_drop_ifg_index(drop_raw: Any) -> np.ndarray:
    """Parse drop_ifg_index from config into a 1D int64 array.

    Handles None, empty list/array, and values from MATLAB v7.3 that may
    come as bytes or object arrays (e.g. '\\x00\\x00...' causing int() to fail).
    """
    if drop_raw is None:
        return np.array([], dtype=np.int64)
    if isinstance(drop_raw, (list, np.ndarray)):
        if len(drop_raw) == 0:
            return np.array([], dtype=np.int64)
        arr = np.asarray(drop_raw)
        if arr.dtype.kind in ("U", "S", "O") or arr.dtype == object:
            try:
                out = np.atleast_1d(np.asarray(arr.flat, dtype=np.float64).astype(np.int64))
            except (ValueError, TypeError):
                return np.array([], dtype=np.int64)
            return np.array([], dtype=np.int64) if out.size == 0 or np.all(out == 0) else out
        try:
            out = np.atleast_1d(np.asarray(arr, dtype=np.int64).ravel())
        except (ValueError, TypeError):
            return np.array([], dtype=np.int64)
        return np.array([], dtype=np.int64) if out.size == 0 or np.all(out == 0) else out
    if isinstance(drop_raw, (int, float, np.integer, np.floating)):
        val = int(drop_raw)
        return np.array([], dtype=np.int64) if val == 0 else np.array([val], dtype=np.int64)
    try:
        out = np.atleast_1d(np.asarray(drop_raw, dtype=np.int64))
    except (ValueError, TypeError):
        return np.array([], dtype=np.int64)
    return np.array([], dtype=np.int64) if out.size == 0 or np.all(out == 0) else out


def _ensure_complex64(arr: np.ndarray) -> np.ndarray:
    """Ensure array is numpy complex64; HDF5 can return compound dtype (real, imag)."""
    if arr.dtype == np.complex64:
        return arr
    if hasattr(arr.dtype, "names") and arr.dtype.names is not None:
        names = arr.dtype.names or ()
        if "real" in names and "imag" in names:
            return (arr["real"] + 1j * arr["imag"]).astype(np.complex64)
        if "r" in names and "i" in names:
            return (arr["r"] + 1j * arr["i"]).astype(np.complex64)
    return np.asarray(arr, dtype=np.complex64)


# ============================================================================
# Delaunay edge extraction helper
# ============================================================================

def _delaunay_edges(xy: np.ndarray) -> np.ndarray:
    """Extract unique edges from a Delaunay triangulation.

    Parameters
    ----------
    xy : (n, 2) float64
        2-D point coordinates.

    Returns
    -------
    edges : (n_edge, 2) int
        Pairs of 0-based point indices.

    MATLAB correspondence (ps_weed.m lines 328-332)::

        tri = delaunay(xy_weed(:,2), xy_weed(:,3));
        tr  = triangulation(tri, xy_weed(:,2), xy_weed(:,3));
        edgs = edges(tr);
    """
    tri = Delaunay(xy)
    # Extract unique edges from simplices
    simplices = tri.simplices  # (n_tri, 3)
    # Each triangle has 3 edges
    edge_pairs = np.vstack([
        simplices[:, [0, 1]],
        simplices[:, [0, 2]],
        simplices[:, [1, 2]],
    ])
    # Sort each pair so (min, max) and unique-ify
    edge_pairs = np.sort(edge_pairs, axis=1)
    edges = np.unique(edge_pairs, axis=0)
    return edges


# ============================================================================
# HDF5 persistence helpers
# ============================================================================

def _save_weed_h5(
    h5_path: Path,
    *,
    ix_weed: np.ndarray,
    ix_weed2: np.ndarray,
    ps_std: np.ndarray,
    ps_max: np.ndarray,
    ifg_index: np.ndarray,
) -> None:
    """Persist ``weed1.h5``.

    MATLAB correspondence (ps_weed.m line 427)::

        stamps_save(weedname, ix_weed, ix_weed2, ps_std, ps_max, ifg_index)
    """
    # MATLAB stores ix_weed, ix_weed2, ps_std, ps_max as (N,1) column vectors,
    # and ifg_index as (1,M) row vector.  We replicate this layout so that
    # downstream consumers see identical shapes.
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("ix_weed", data=ix_weed.ravel()[:, None].astype(np.uint8))
        hf.create_dataset("ix_weed2", data=ix_weed2.ravel()[:, None].astype(np.uint8))
        hf.create_dataset("ps_std", data=ps_std.ravel()[:, None].astype(np.float32))
        hf.create_dataset("ps_max", data=ps_max.ravel()[:, None].astype(np.float32))
        hf.create_dataset("ifg_index", data=ifg_index.ravel()[None, :].astype(np.uint16))
    logger.info("Saved %s", h5_path.name)


def _save_ps2_h5(
    h5_path: Path,
    *,
    bperp: np.ndarray,
    day: np.ndarray,
    day_ix: np.ndarray,
    master_day: int,
    master_ix: int,
    n_ifg: int,
    n_image: int,
    n_ps: int,
    ij: np.ndarray,
    lonlat: np.ndarray,
    xy: np.ndarray,
    ifgday: Optional[np.ndarray] = None,
    ifgday_ix: Optional[np.ndarray] = None,
    ll0: Optional[np.ndarray] = None,
) -> None:
    """Persist ``ps2.h5`` — the weeded PS metadata.

    MATLAB correspondence (ps_weed.m line 461)::

        save(psname, '-struct', ps)
    """
    # MATLAB dimension conventions:
    #   bperp (N,1), day (M,1), day_ix (M-1,1), scalars (1,1),
    #   ij (n_ps,3), lonlat (n_ps,2), xy (n_ps,3),
    #   ifgday (N,2), ifgday_ix (N,2), ll0 (1,2).
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("bperp", data=bperp.ravel()[:, None])
        hf.create_dataset("day", data=day.ravel()[:, None].astype(np.int32))
        hf.create_dataset("day_ix", data=_as_matlab_index_vector(day_ix)[:, None])
        hf.create_dataset("master_day", data=np.array([[master_day]], dtype=np.int32))
        hf.create_dataset("master_ix", data=np.array([[master_ix]], dtype=np.int32))
        hf.create_dataset("n_ifg", data=np.array([[n_ifg]], dtype=np.int32))
        hf.create_dataset("n_image", data=np.array([[n_image]], dtype=np.int32))
        hf.create_dataset("n_ps", data=np.array([[n_ps]], dtype=np.int32))
        hf.create_dataset("ij", data=ij)
        hf.create_dataset("lonlat", data=lonlat)
        hf.create_dataset("xy", data=xy.astype(np.float32))
        if ifgday is not None:
            hf.create_dataset("ifgday", data=ifgday)
        if ifgday_ix is not None:
            hf.create_dataset("ifgday_ix", data=ifgday_ix)
        if ll0 is not None:
            ll0_2d = np.atleast_2d(ll0)
            if ll0_2d.shape[0] > ll0_2d.shape[1]:
                ll0_2d = ll0_2d.T  # ensure (1, 2)
            hf.create_dataset("ll0", data=ll0_2d)
    logger.info("Saved %s (n_ps=%d)", h5_path.name, n_ps)


def _save_pm2_h5(
    h5_path: Path,
    *,
    ph_patch: np.ndarray,
    ph_res: np.ndarray,
    coh_ps: np.ndarray,
    K_ps: np.ndarray,
    C_ps: np.ndarray,
) -> None:
    """Persist ``pm2.h5``.

    MATLAB correspondence (ps_weed.m line 441)::

        stamps_save(pmname, ph_patch, ph_res, coh_ps, K_ps, C_ps)
    """
    # MATLAB stores coh_ps, K_ps, C_ps as (N,1) column vectors.
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("ph_patch", data=ph_patch.astype(np.complex64), compression="gzip")
        hf.create_dataset("ph_res", data=ph_res.astype(np.float32), compression="gzip")
        hf.create_dataset("coh_ps", data=coh_ps.ravel()[:, None].astype(np.float64))
        hf.create_dataset("K_ps", data=K_ps.ravel()[:, None].astype(np.float64))
        hf.create_dataset("C_ps", data=C_ps.ravel()[:, None].astype(np.float64))
    logger.info("Saved %s", h5_path.name)


def _save_simple_h5(h5_path: Path, name: str, data: np.ndarray,
                    column_vector: bool = False) -> None:
    """Save a single array to an HDF5 file (ph2, hgt2, bp2, inc2).

    Parameters
    ----------
    column_vector : bool
        If True, reshape 1-D data to ``(N, 1)`` to match MATLAB column-vector
        convention (used for ``hgt``, ``inc``).
    """
    if column_vector and data.ndim == 1:
        data = data[:, None]
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset(name, data=data,
                          compression="gzip" if data.nbytes > 1_000_000 else None)
    logger.info("Saved %s (/%s)", h5_path.name, name)


# ============================================================================
# Main weeder class
# ============================================================================

class PSWeeder:
    """Weed out neighbouring and noisy PS — port of ``ps_weed.m``.

    Parameters
    ----------
    patch_dir : Path
        Directory containing Step-1/2/3 outputs.
    psver : int
        PS version number (default 1).
    """

    def __init__(self, patch_dir: Union[str, Path], psver: int = 1) -> None:
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver

        self._cfg = StampsConfig(work_dir=self.patch_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        all_da_flag: int = 0,
        no_weed_adjacent: Optional[int] = None,
        no_weed_noisy: Optional[int] = None,
    ) -> None:
        """Execute PS weeding.

        Parameters
        ----------
        all_da_flag : int
            0 = normal (default); ≠0 = include high-D_A PS (not implemented).
        no_weed_adjacent : int or None
            0 = weed adjacent, 1 = skip.
            None → determined by ``weed_neighbours`` parm.
        no_weed_noisy : int or None
            0 = weed noisy, 1 = skip.
            None → determined by ``weed_standard_dev`` / ``weed_max_noise``.

        MATLAB correspondence (ps_weed.m lines 31-57).
        """
        t_start = time.time()
        logger.info("Weeding selected pixels...")

        # --- Load parameters ---
        parms = self._load_parameters()
        weed_std = parms["weed_standard_dev"]
        weed_max = parms["weed_max_noise"]
        weed_zero_elev = parms["weed_zero_elevation"]
        small_baseline_flag = parms["small_baseline_flag"]
        drop_ifg_index = parms["drop_ifg_index"]
        time_win = parms["weed_time_win"]

        # Determine weed flags (lines 44-57)
        if no_weed_adjacent is None:
            if parms["weed_neighbours"].lower() == "y":
                no_weed_adjacent = 0
            else:
                no_weed_adjacent = 1

        if no_weed_noisy is None:
            if weed_std >= np.pi and weed_max >= np.pi:
                no_weed_noisy = 1
            else:
                no_weed_noisy = 0

        # --- Load data ---
        (ps_data, ph, sl_data, pm_data,
         bperp_1d, day, master_day, master_ix,
         n_ifg_orig) = self._load_all_data(parms)

        # --- ifg_index (line 78) ---
        all_ifg_1 = np.arange(1, n_ifg_orig + 1)
        if len(drop_ifg_index) > 0:
            ifg_index_1 = np.setdiff1d(all_ifg_1, drop_ifg_index)
        else:
            ifg_index_1 = all_ifg_1.copy()
        ifg_index_0 = ifg_index_1.astype(np.int64) - 1  # 0-based

        # --- Extract kept PS from Step 3 (lines 94-104) ---
        # MATLAB: ix2 = sl.ix(sl.keep_ix)  (1-based)
        sl_ix = sl_data["ix"]       # 1-based
        sl_keep = sl_data["keep_ix"].astype(bool)
        ix2_1 = sl_ix[sl_keep]      # 1-based into ps1 arrays
        ix2_0 = ix2_1 - 1           # 0-based

        K_ps2 = sl_data["K_ps2"][sl_keep]
        C_ps2 = sl_data["C_ps2"][sl_keep]
        coh_ps2 = sl_data["coh_ps2"][sl_keep]

        ij2 = ps_data["ij"][ix2_0]
        xy2 = ps_data["xy"][ix2_0]
        ph2 = ph[ix2_0]
        lonlat2 = ps_data["lonlat"][ix2_0]

        # ph_patch from pm (original, with PS left in) — line 112
        ph_patch2 = pm_data["ph_patch"][ix2_0]

        # ph_res2 from select — line 113-117
        if "ph_res2" in sl_data and sl_data["ph_res2"] is not None:
            ph_res2 = sl_data["ph_res2"][sl_keep]
        else:
            ph_res2 = np.array([], dtype=np.float32)

        # hgt — line 156-164
        hgt = self._load_hgt(ix2_0)

        # inc — line 487-498
        inc = self._load_inc(ix2_0)

        n_ps_low_DA = len(ix2_0)
        n_ps_other = 0  # all_da_flag == 0
        n_ps = n_ps_low_DA + n_ps_other
        ix_weed = np.ones(n_ps, dtype=bool)

        logger.info("%d low-D_A PS, %d high-D_A PS", n_ps_low_DA, n_ps_other)

        # ================================================================
        # 1. Weed adjacent pixels (lines 176-246)
        # ================================================================
        if no_weed_adjacent == 0:
            ix_weed = self._weed_adjacent(ij2, coh_ps2, ix_weed)
            logger.info("%d PS kept after dropping adjacent pixels", int(ix_weed.sum()))

        # ================================================================
        # 2. Weed zero elevation (lines 250-254)
        # ================================================================
        if weed_zero_elev.lower() == "y" and hgt is not None:
            sea_ix = hgt < 1e-6
            ix_weed[sea_ix] = False
            logger.info("%d PS kept after weeding zero elevation", int(ix_weed.sum()))

        xy_weed = xy2[ix_weed]
        n_ps = int(ix_weed.sum())

        # ================================================================
        # 3. Remove duplicate lon/lat (lines 259-276)
        # ================================================================
        ix_weed = self._remove_duplicates(xy_weed, coh_ps2, ix_weed)
        xy_weed = xy2[ix_weed]
        n_ps = int(ix_weed.sum())
        logger.info("%d PS after duplicate removal", n_ps)

        # ================================================================
        # 4. Weed noisy pixels (lines 283-403)
        # ================================================================
        ix_weed2 = np.ones(n_ps, dtype=bool)
        ps_std = np.zeros(n_ps, dtype=np.float32)
        ps_max = np.zeros(n_ps, dtype=np.float32)

        if n_ps > 0 and no_weed_noisy == 0:
            ix_weed2, ps_std, ps_max = self._weed_noisy(
                ph2, ix_weed, K_ps2, C_ps2, bperp_1d,
                small_baseline_flag, master_ix, day,
                ifg_index_0, drop_ifg_index, time_win,
                xy2, weed_std, weed_max,
            )
            ix_weed[ix_weed] = ix_weed2
            n_ps = int(ix_weed.sum())
            logger.info("%d PS kept after dropping noisy pixels", n_ps)

        # ================================================================
        # Update no_ps_info (lines 408-421)
        # ================================================================
        no_ps_path = self.patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = self._load_no_ps_info(no_ps_path)
        stamps_step_no_ps[3:] = 0
        if n_ps == 0:
            logger.warning("*** No PS points left after weeding ***")
            stamps_step_no_ps[3] = 1
        self._save_no_ps_info(no_ps_path, stamps_step_no_ps)

        # ================================================================
        # Save weed1 (line 427)
        # ================================================================
        weed_path = self.patch_dir / f"weed{self.psver}.h5"
        _save_weed_h5(
            weed_path,
            ix_weed=ix_weed,
            ix_weed2=ix_weed2,
            ps_std=ps_std,
            ps_max=ps_max,
            ifg_index=ifg_index_1,  # 1-based
        )

        # ================================================================
        # Save version 2 files (lines 429-532)
        # ================================================================
        self._save_version2(
            ix_weed, ph2, xy2, ij2, lonlat2,
            coh_ps2, K_ps2, C_ps2, ph_patch2, ph_res2,
            ps_data, hgt, inc,
            bperp_1d, ix2_0,
        )

        # Write psver=2 marker
        psver_path = self.patch_dir / "psver.h5"
        with h5py.File(str(psver_path), "w") as hf:
            hf.create_dataset("psver", data=np.int32(self.psver + 1))
        logger.info("Updated psver to %d", self.psver + 1)

        elapsed = time.time() - t_start
        logger.info(
            "Step 4 complete: %d → %d PS (%.1f s)",
            n_ps_low_DA, n_ps, elapsed,
        )

    # ------------------------------------------------------------------
    # Internal: load parameters
    # ------------------------------------------------------------------
    def _load_parameters(self) -> dict:
        """Read weeding parameters from config.

        MATLAB correspondence (ps_weed.m lines 36-42).
        """
        cfg = self._cfg
        drop_ifg_index = _parse_drop_ifg_index(cfg.getparm("drop_ifg_index"))

        return {
            "weed_time_win": float(cfg.getparm("weed_time_win")),
            "weed_standard_dev": float(cfg.getparm("weed_standard_dev")),
            "weed_max_noise": float(cfg.getparm("weed_max_noise")),
            "weed_zero_elevation": str(cfg.getparm("weed_zero_elevation")).strip(),
            "weed_neighbours": str(cfg.getparm("weed_neighbours")).strip(),
            "small_baseline_flag": str(cfg.getparm("small_baseline_flag")).strip().lower(),
            "drop_ifg_index": drop_ifg_index,
        }

    # ------------------------------------------------------------------
    # Internal: load all input data
    # ------------------------------------------------------------------
    def _load_all_data(self, parms: dict):
        """Load ps1, ph1, select1, pm1 data."""
        v = self.psver

        # ps1
        ps_h5 = self.patch_dir / f"ps{v}.h5"
        ps_mat = self.patch_dir / f"ps{v}.mat"
        if ps_h5.is_file():
            with h5py.File(str(ps_h5), "r") as hf:
                ps_data = {
                    "ij": np.asarray(hf["ij"][:]),
                    "xy": np.asarray(hf["xy"][:], dtype=np.float32),
                    "lonlat": np.asarray(hf["lonlat"][:]),
                    "n_ifg": int(hf["n_ifg"][()]),
                    "n_image": int(hf["n_image"][()]),
                    "master_ix": int(hf["master_ix"][()]),
                    "master_day": int(hf["master_day"][()]),
                    "bperp": np.asarray(hf["bperp"][:], dtype=np.float64).ravel(),
                    "day": np.asarray(hf["day"][:]).ravel(),
                    "day_ix": np.asarray(hf["day_ix"][:]).ravel(),
                }
                if "ifgday" in hf:
                    ps_data["ifgday"] = np.asarray(hf["ifgday"][:])
                if "ifgday_ix" in hf:
                    ps_data["ifgday_ix"] = np.asarray(hf["ifgday_ix"][:])
                if "ll0" in hf:
                    ps_data["ll0"] = np.asarray(hf["ll0"][:])
        elif ps_mat.is_file():
            d = load_mat(ps_mat)
            ps_data = {
                "ij": np.asarray(d["ij"]),
                "xy": np.asarray(d["xy"]).astype(np.float32),
                "lonlat": np.asarray(d["lonlat"]),
                "n_ifg": int(np.asarray(d["n_ifg"]).ravel()[0]),
                "n_image": int(np.asarray(d["n_image"]).ravel()[0]),
                "master_ix": int(np.asarray(d["master_ix"]).ravel()[0]),
                "master_day": int(np.asarray(d["master_day"]).ravel()[0]),
                "bperp": np.asarray(d["bperp"]).ravel().astype(np.float64),
                "day": np.asarray(d["day"]).ravel(),
                "day_ix": np.asarray(d["day_ix"]).ravel(),
            }
            if "ifgday" in d:
                ps_data["ifgday"] = np.asarray(d["ifgday"])
            if "ifgday_ix" in d:
                ps_data["ifgday_ix"] = np.asarray(d["ifgday_ix"])
            if "ll0" in d:
                ps_data["ll0"] = np.asarray(d["ll0"])
        else:
            raise FileNotFoundError(f"ps{v} not found in {self.patch_dir}")

        # ph
        ph = self._load_ph()

        # select1
        sl_h5 = self.patch_dir / f"select{v}.h5"
        sl_mat = self.patch_dir / f"select{v}.mat"
        if sl_h5.is_file():
            with h5py.File(str(sl_h5), "r") as hf:
                sl_data = {
                    "ix": np.asarray(hf["ix"][:]).ravel().astype(np.int64),
                    "keep_ix": np.asarray(hf["keep_ix"][:]).ravel(),
                    "K_ps2": np.asarray(hf["K_ps2"][:]).ravel().astype(np.float64),
                    "C_ps2": np.asarray(hf["C_ps2"][:]).ravel().astype(np.float64),
                    "coh_ps2": np.asarray(hf["coh_ps2"][:]).ravel().astype(np.float64),
                }
                if "ph_res2" in hf:
                    sl_data["ph_res2"] = np.asarray(hf["ph_res2"][:])
                else:
                    sl_data["ph_res2"] = None
        elif sl_mat.is_file():
            d = load_mat(sl_mat)
            sl_data = {
                "ix": np.asarray(d["ix"]).ravel().astype(np.int64),
                "keep_ix": np.asarray(d["keep_ix"]).ravel(),
                "K_ps2": np.asarray(d["K_ps2"]).ravel().astype(np.float64),
                "C_ps2": np.asarray(d["C_ps2"]).ravel().astype(np.float64),
                "coh_ps2": np.asarray(d["coh_ps2"]).ravel().astype(np.float64),
            }
            if "ph_res2" in d:
                sl_data["ph_res2"] = d["ph_res2"]
            else:
                sl_data["ph_res2"] = None
        else:
            raise FileNotFoundError(f"select{v} not found in {self.patch_dir}")

        # pm1
        pm_h5 = self.patch_dir / f"pm{v}.h5"
        pm_mat = self.patch_dir / f"pm{v}.mat"
        if pm_h5.is_file():
            with h5py.File(str(pm_h5), "r") as hf:
                pm_data = {"ph_patch": np.asarray(hf["ph_patch"][:])}
        elif pm_mat.is_file():
            d = load_mat(pm_mat)
            pm_data = {"ph_patch": np.asarray(d["ph_patch"])}
        else:
            raise FileNotFoundError(f"pm{v} not found in {self.patch_dir}")

        bperp_1d = ps_data["bperp"]
        day = ps_data["day"]
        master_day = ps_data["master_day"]
        master_ix = ps_data["master_ix"]
        n_ifg = ps_data["n_ifg"]

        return (ps_data, ph, sl_data, pm_data,
                bperp_1d, day, master_day, master_ix, n_ifg)

    def _load_ph(self) -> np.ndarray:
        """Load ph from ph1 or ps1."""
        v = self.psver
        ph_h5 = self.patch_dir / f"ph{v}.h5"
        if ph_h5.is_file():
            with h5py.File(str(ph_h5), "r") as hf:
                return np.asarray(hf["ph"][:])
        ph_mat = self.patch_dir / f"ph{v}.mat"
        if ph_mat.is_file():
            return np.asarray(load_mat(ph_mat)["ph"])
        ps_h5 = self.patch_dir / f"ps{v}.h5"
        with h5py.File(str(ps_h5), "r") as hf:
            if "ph" in hf:
                return np.asarray(hf["ph"][:])
        raise FileNotFoundError(f"ph data not found in {self.patch_dir}")

    def _load_hgt(self, ix: np.ndarray) -> Optional[np.ndarray]:
        """Load height data for selected pixels."""
        v = self.psver
        for path, key in [
            (self.patch_dir / f"hgt{v}.h5", "hgt"),
            (self.patch_dir / f"hgt{v}.mat", "hgt"),
        ]:
            if path.is_file():
                if path.suffix == ".h5":
                    with h5py.File(str(path), "r") as hf:
                        return np.asarray(hf[key][:]).ravel()[ix]
                else:
                    return np.asarray(load_mat(path)[key]).ravel()[ix]
        # Check ps1.h5
        ps_h5 = self.patch_dir / f"ps{v}.h5"
        if ps_h5.is_file():
            with h5py.File(str(ps_h5), "r") as hf:
                if "hgt" in hf:
                    return np.asarray(hf["hgt"][:]).ravel()[ix]
        return None

    def _load_inc(self, ix: np.ndarray) -> Optional[np.ndarray]:
        """Load incidence angle for selected pixels."""
        v = self.psver
        for path, key in [
            (self.patch_dir / f"inc{v}.h5", "inc"),
            (self.patch_dir / f"inc{v}.mat", "inc"),
        ]:
            if path.is_file():
                if path.suffix == ".h5":
                    with h5py.File(str(path), "r") as hf:
                        return np.asarray(hf[key][:]).ravel()[ix]
                else:
                    return np.asarray(load_mat(path)[key]).ravel()[ix]
        return None

    # ------------------------------------------------------------------
    # 1. Weed adjacent pixels
    # ------------------------------------------------------------------
    def _weed_adjacent(
        self,
        ij2: np.ndarray,
        coh_ps2: np.ndarray,
        ix_weed: np.ndarray,
    ) -> np.ndarray:
        """Remove adjacent PS, keeping the one with highest coherence.

        Uses a grid-based approach matching MATLAB's ``neigh_ix`` logic
        (ps_weed.m lines 176-246), but implemented efficiently using
        NumPy array operations.

        MATLAB algorithm
        ----------------
        1. Build a 2-D grid ``neigh_ix`` where each cell stores the index
           of the last PS that claimed it as a neighbour.
        2. For each PS, look at its cell in ``neigh_ix``: if another PS
           already claimed it, they are neighbours.
        3. BFS-merge connected neighbour groups; keep the highest-coh one.

        Python implementation
        ---------------------
        We replicate the same grid-based approach: shift ij coordinates,
        build a sparse grid, and find groups of pixels sharing the same
        3×3 neighbourhood.
        """
        logger.info("Weeding adjacent pixels...")
        n_ps = len(ij2)

        # Shift ij to ensure positive indices with margin (MATLAB line 180)
        ij_shift = ij2[:, 1:3].astype(np.int64)  # (n_ps, 2): [az, rg]
        ij_shift = ij_shift - ij_shift.min(axis=0) + 2  # +2 for 1-cell border

        max_i = ij_shift[:, 0].max() + 2
        max_j = ij_shift[:, 1].max() + 2

        # Build neighbour grid (MATLAB lines 181-195)
        # neigh_ix stores the "owner" PS index for each cell
        neigh_ix = np.zeros((max_i, max_j), dtype=np.int64)
        # Store which PS claims which cells; build neighbour lists
        neigh_ps = [[] for _ in range(n_ps)]

        for i in range(n_ps):
            ci, cj = ij_shift[i, 0], ij_shift[i, 1]
            # Check 3×3 neighbourhood
            patch = neigh_ix[ci - 1: ci + 2, cj - 1: cj + 2]  # (3,3)
            # Mark unclaimed cells (except centre) with this pixel
            miss_middle = np.ones((3, 3), dtype=bool)
            miss_middle[1, 1] = False
            mask = (patch == 0) & miss_middle
            patch[mask] = i + 1  # 1-based for storage (0 = empty)
            neigh_ix[ci - 1: ci + 2, cj - 1: cj + 2] = patch

        # Find neighbours (MATLAB lines 201-213)
        for i in range(n_ps):
            ci, cj = ij_shift[i, 0], ij_shift[i, 1]
            owner = neigh_ix[ci, cj]
            if owner != 0 and owner != i + 1:
                neigh_ps[owner - 1].append(i)

        # Select best from each group (MATLAB lines 218-243)
        for i in range(n_ps):
            if neigh_ps[i]:
                # BFS to merge connected groups
                same_ps = [i]
                visited = set()
                visited.add(i)
                queue = list(neigh_ps[i])
                neigh_ps[i] = []
                while queue:
                    ps_i = queue.pop(0)
                    if ps_i in visited:
                        continue
                    visited.add(ps_i)
                    same_ps.append(ps_i)
                    queue.extend(neigh_ps[ps_i])
                    neigh_ps[ps_i] = []

                same_ps = list(set(same_ps))
                if len(same_ps) > 1:
                    cohs = coh_ps2[same_ps]
                    best = same_ps[np.argmax(cohs)]
                    for p in same_ps:
                        if p != best:
                            ix_weed[p] = False

        return ix_weed

    # ------------------------------------------------------------------
    # 3. Remove duplicate xy coordinates
    # ------------------------------------------------------------------
    def _remove_duplicates(
        self,
        xy_weed: np.ndarray,
        coh_ps2: np.ndarray,
        ix_weed: np.ndarray,
    ) -> np.ndarray:
        """Remove pixels with duplicate xy, keeping highest coherence.

        MATLAB correspondence (ps_weed.m lines 259-276)::

            [~, I] = unique(xy_weed(:,2:3), 'rows');
            dups = setxor(I, [1:sum(ix_weed)]');
            ...keep highest coh among duplicates...
        """
        n_weed = int(ix_weed.sum())
        if n_weed == 0:
            return ix_weed

        ix_weed_num = np.where(ix_weed)[0]

        _, unique_idx = np.unique(xy_weed[:, 1:3], axis=0, return_index=True)
        all_idx = np.arange(n_weed)
        dup_idx = np.setdiff1d(all_idx, unique_idx)

        if len(dup_idx) == 0:
            return ix_weed

        n_dropped = 0
        for di in dup_idx:
            # Find all points with same xy
            same = np.where(
                (xy_weed[:, 1] == xy_weed[di, 1]) &
                (xy_weed[:, 2] == xy_weed[di, 2])
            )[0]
            orig_idx = ix_weed_num[same]
            best = orig_idx[np.argmax(coh_ps2[orig_idx])]
            for oi in orig_idx:
                if oi != best:
                    if ix_weed[oi]:
                        ix_weed[oi] = False
                        n_dropped += 1

        if n_dropped > 0:
            logger.info("%d PS with duplicate lon/lat dropped", n_dropped)
        return ix_weed

    # ------------------------------------------------------------------
    # 4. Weed noisy pixels
    # ------------------------------------------------------------------
    def _weed_noisy(
        self,
        ph2: np.ndarray,
        ix_weed: np.ndarray,
        K_ps2: np.ndarray,
        C_ps2: np.ndarray,
        bperp: np.ndarray,
        small_baseline_flag: str,
        master_ix: int,
        day: np.ndarray,
        ifg_index_0: np.ndarray,
        drop_ifg_index: np.ndarray,
        time_win: float,
        xy2: np.ndarray,
        weed_std: float,
        weed_max: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weed noisy pixels using Delaunay triangulation.

        MATLAB correspondence (ps_weed.m lines 283-403).

        Returns
        -------
        ix_weed2 : bool array for the weeded subset
        ps_std : float32 array, per-pixel min edge std
        ps_max : float32 array, per-pixel min edge max
        """
        logger.info("Weeding noisy pixels via Delaunay triangulation...")

        n_ps = int(ix_weed.sum())
        xy_weed = xy2[ix_weed].astype(np.float64)

        # Build Delaunay triangulation (line 329-332)
        edges = _delaunay_edges(xy_weed[:, 1:3])
        n_edge = len(edges)
        logger.info("Delaunay: %d edges from %d points", n_edge, n_ps)

        # Corrected phase (line 335-339)
        # ph_weed = ph2(ix_weed,:) .* exp(-j*(K_ps2(ix_weed)*bperp'))
        ph2_slice = _ensure_complex64(ph2[ix_weed])
        ph_weed = ph2_slice * np.exp(-1j * K_ps2[ix_weed, None] * bperp[None, :])
        abs_ph = np.abs(ph_weed)
        abs_ph[abs_ph == 0] = 1.0
        ph_weed = ph_weed / abs_ph

        if small_baseline_flag != "y":
            # Add master noise (line 338)
            midx_0 = int(master_ix) - 1
            ph_weed[:, midx_0] = np.exp(1j * C_ps2[ix_weed])

        # Arc differential phase (line 342-343)
        dph_space = ph_weed[edges[:, 1]] * np.conj(ph_weed[edges[:, 0]])
        dph_space = dph_space[:, ifg_index_0]
        n_use = len(ifg_index_0)

        bperp_used = bperp[ifg_index_0]

        if small_baseline_flag != "y":
            # PS mode: time-domain smoothing (lines 354-380)
            edge_std, edge_max = self._noise_ps_mode(
                dph_space, day, ifg_index_0, time_win, bperp_used, n_edge, n_use,
            )
        else:
            # SB mode: simpler noise estimation (lines 382-387)
            edge_std, edge_max = self._noise_sb_mode(
                dph_space, bperp_used, n_edge, n_use,
            )

        # Per-pixel minimum edge noise (lines 392-397)
        logger.info("Estimating max noise for all pixels...")
        ps_std = np.full(n_ps, np.inf, dtype=np.float32)
        ps_max = np.full(n_ps, np.inf, dtype=np.float32)

        # Vectorised: for each edge, update both endpoints
        e1 = edges[:, 0]
        e2 = edges[:, 1]
        # Use np.minimum.at for scatter-reduce
        np.minimum.at(ps_std, e1, edge_std)
        np.minimum.at(ps_std, e2, edge_std)
        np.minimum.at(ps_max, e1, edge_max)
        np.minimum.at(ps_max, e2, edge_max)

        # Threshold (line 398)
        ix_weed2 = (ps_std < weed_std) & (ps_max < weed_max)

        return ix_weed2, ps_std, ps_max

    def _noise_sb_mode(
        self,
        dph_space: np.ndarray,
        bperp_used: np.ndarray,
        n_edge: int,
        n_use: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SB-mode arc noise estimation.

        MATLAB correspondence (ps_weed.m lines 382-387)::

            ifg_var = var(dph_space, 0, 1);
            K = lscov(bperp(ifg_index), dph_space', 1./ifg_var)';
            dph_space = dph_space - K * bperp(ifg_index)';
            edge_std = std(angle(dph_space), 0, 2);
            edge_max = max(abs(angle(dph_space)), [], 2);
        """
        # Variance per IFG (MATLAB: var(…,0,1) = population variance along axis=0)
        # But MATLAB var(X,0,1) with flag=0 uses N-1 denominator
        ifg_var = np.var(dph_space, axis=0, ddof=1)
        ifg_var[ifg_var == 0] = 1.0

        # Weighted least-squares: K = lscov(bperp, dph_space', 1./ifg_var)
        # lscov(A, B, w) solves WLS: min || sqrt(W) * (A*x - b) ||^2
        # For each edge: K_edge = (bperp' W bperp)^-1 bperp' W dph_edge
        W = 1.0 / ifg_var  # (n_use,)
        # Vectorised: A = bperp (n_use, 1), B = dph_space.T (n_use, n_edge)
        A = bperp_used[:, None]  # (n_use, 1)
        AW = A * W[:, None]     # (n_use, 1) weighted
        denom = (AW * A).sum(axis=0)  # scalar
        if denom == 0:
            denom = 1.0
        # K = (A'WA)^-1 A'W * dph_space.T
        K = (AW[:, 0] @ dph_space) / denom  # (n_edge,)

        dph_corrected = dph_space - K[:, None] * bperp_used[None, :]
        dph_angle = np.angle(dph_corrected)

        edge_std = np.std(dph_angle, axis=1, ddof=1).astype(np.float32)
        edge_max = np.max(np.abs(dph_angle), axis=1).astype(np.float32)

        return edge_std, edge_max

    def _noise_ps_mode(
        self,
        dph_space: np.ndarray,
        day: np.ndarray,
        ifg_index_0: np.ndarray,
        time_win: float,
        bperp_used: np.ndarray,
        n_edge: int,
        n_use: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PS-mode arc noise estimation with time-domain smoothing.

        MATLAB correspondence (ps_weed.m lines 354-380).
        """
        day_used = day[ifg_index_0].astype(np.float64)

        dph_smooth = np.zeros((n_edge, n_use), dtype=np.complex64)
        dph_smooth2 = np.zeros((n_edge, n_use), dtype=np.complex64)

        for i1 in range(n_use):
            time_diff = day_used[i1] - day_used
            weight_factor = np.exp(-(time_diff ** 2) / (2 * time_win ** 2))
            weight_factor /= weight_factor.sum()

            dph_mean = np.sum(
                dph_space * weight_factor[None, :], axis=1
            )  # (n_edge,)
            dph_mean_adj = np.angle(
                dph_space * np.conj(dph_mean[:, None])
            )  # (n_edge, n_use)

            G = np.column_stack([np.ones(n_use), time_diff])  # (n_use, 2)

            # Weighted least-squares per edge (vectorised)
            Gw = G * weight_factor[:, None]
            GwG_inv = np.linalg.pinv(Gw.T @ G)
            m = (GwG_inv @ Gw.T @ dph_mean_adj.T)  # (2, n_edge)

            dph_mean_adj2 = np.angle(
                np.exp(1j * (dph_mean_adj - (G @ m).T))
            )
            m2 = (GwG_inv @ Gw.T @ dph_mean_adj2.T)

            dph_smooth[:, i1] = dph_mean * np.exp(
                1j * (m[0, :] + m2[0, :])
            )

            wf2 = weight_factor.copy()
            wf2[i1] = 0
            dph_smooth2[:, i1] = np.sum(
                dph_space * wf2[None, :], axis=1
            )

        dph_noise = np.angle(dph_space * np.conj(dph_smooth))
        dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
        ifg_var = np.var(dph_noise2, axis=0, ddof=1)
        ifg_var[ifg_var == 0] = 1.0

        # DEM error estimation (MATLAB: K = lscov(bperp(ifg_index), double(dph_noise)', 1./ifg_var)')
        # dph_noise is (n_edge, n_use); lscov expects B = (n_use, n_edge), so we use dph_noise.T
        W = 1.0 / ifg_var
        A = bperp_used[:, None]
        AW = A * W[:, None]
        denom = float((AW * A).sum())
        if denom == 0:
            denom = 1.0
        K = (AW[:, 0] @ dph_noise.T) / denom  # (n_use,) @ (n_use, n_edge) -> (n_edge,)

        dph_noise = dph_noise - K[:, None] * bperp_used[None, :]

        edge_std = np.std(dph_noise, axis=1, ddof=1).astype(np.float32)
        edge_max = np.max(np.abs(dph_noise), axis=1).astype(np.float32)

        return edge_std, edge_max

    # ------------------------------------------------------------------
    # Save version 2 files
    # ------------------------------------------------------------------
    def _save_version2(
        self,
        ix_weed: np.ndarray,
        ph2: np.ndarray,
        xy2: np.ndarray,
        ij2: np.ndarray,
        lonlat2: np.ndarray,
        coh_ps2: np.ndarray,
        K_ps2: np.ndarray,
        C_ps2: np.ndarray,
        ph_patch2: np.ndarray,
        ph_res2: np.ndarray,
        ps_data: dict,
        hgt: Optional[np.ndarray],
        inc: Optional[np.ndarray],
        bperp_1d: np.ndarray,
        ix2_0: np.ndarray,
    ) -> None:
        """Save all version-2 output files.

        MATLAB correspondence (ps_weed.m lines 429-512).
        """
        v2 = self.psver + 1
        d = self.patch_dir

        # pm2 (line 441)
        _save_pm2_h5(
            d / f"pm{v2}.h5",
            ph_patch=ph_patch2[ix_weed],
            ph_res=ph_res2[ix_weed] if len(ph_res2) > 0 else ph_res2,
            coh_ps=coh_ps2[ix_weed],
            K_ps=K_ps2[ix_weed],
            C_ps=C_ps2[ix_weed],
        )

        # ph2 (line 449)
        _save_simple_h5(d / f"ph{v2}.h5", "ph", _ensure_complex64(ph2[ix_weed]))

        # ps2 (line 461)
        n_ps_final = int(ix_weed.sum())
        kw = {
            "bperp": ps_data["bperp"],
            "day": ps_data["day"],
            "day_ix": ps_data["day_ix"],
            "master_day": ps_data["master_day"],
            "master_ix": ps_data["master_ix"],
            "n_ifg": ps_data["n_ifg"],
            "n_image": ps_data["n_image"],
            "n_ps": n_ps_final,
            "ij": ij2[ix_weed],
            "lonlat": lonlat2[ix_weed],
            "xy": xy2[ix_weed],
        }
        if "ifgday" in ps_data:
            kw["ifgday"] = ps_data["ifgday"]
        if "ifgday_ix" in ps_data:
            kw["ifgday_ix"] = ps_data["ifgday_ix"]
        if "ll0" in ps_data:
            kw["ll0"] = ps_data["ll0"]
        _save_ps2_h5(d / f"ps{v2}.h5", **kw)

        # hgt2 (line 466)
        if hgt is not None:
            _save_simple_h5(d / f"hgt{v2}.h5", "hgt", hgt[ix_weed].astype(np.float64),
                           column_vector=True)

        # inc2 (line 496)
        if inc is not None:
            _save_simple_h5(d / f"inc{v2}.h5", "inc", inc[ix_weed].astype(np.float64),
                           column_vector=True)

        # bp2 — bperp_mat (line 510)
        bp_h5 = self.patch_dir / f"bp{self.psver}.h5"
        bp_mat = self.patch_dir / f"bp{self.psver}.mat"
        bperp_mat = None
        if bp_h5.is_file():
            with h5py.File(str(bp_h5), "r") as hf:
                ds = hf["bperp_mat"]
                logger.info("Loading bperp_mat from %s", bp_h5.name)
                if ds.shape[0] == len(ix2_0):
                    bperp_mat = np.asarray(ds, dtype=np.float32)
                else:
                    # h5py fancy indexing over hundreds of thousands of rows is
                    # much slower than one sequential read for this matrix size.
                    bperp_mat = np.asarray(ds, dtype=np.float32)[ix2_0]
        elif bp_mat.is_file():
            bperp_mat = np.asarray(load_mat(bp_mat)["bperp_mat"])[ix2_0].astype(np.float32)
        else:
            # Fall back to ps1.h5
            ps_h5 = self.patch_dir / f"ps{self.psver}.h5"
            if ps_h5.is_file():
                with h5py.File(str(ps_h5), "r") as hf:
                    if "bperp_mat" in hf:
                        ds = hf["bperp_mat"]
                        logger.info("Loading bperp_mat from %s", ps_h5.name)
                        if ds.shape[0] == len(ix2_0):
                            bperp_mat = np.asarray(ds, dtype=np.float32)
                        else:
                            # h5py fancy indexing over hundreds of thousands of
                            # rows is much slower than one sequential read for
                            # this matrix size.
                            bperp_mat = np.asarray(ds, dtype=np.float32)[ix2_0]

        if bperp_mat is not None:
            bperp_mat_weed = bperp_mat[ix_weed]
            _save_simple_h5(d / f"bp{v2}.h5", "bperp_mat", bperp_mat_weed)

    # ------------------------------------------------------------------
    # no_ps_info helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_no_ps_info(h5_path: Path) -> np.ndarray:
        if h5_path.is_file():
            with h5py.File(str(h5_path), "r") as hf:
                return np.array(hf["stamps_step_no_ps"])
        return np.zeros(5, dtype=np.int32)

    @staticmethod
    def _save_no_ps_info(h5_path: Path, flags: np.ndarray) -> None:
        with h5py.File(str(h5_path), "w") as hf:
            hf.create_dataset("stamps_step_no_ps", data=flags)


# ============================================================================
# Module-level convenience function
# ============================================================================

def ps_weed(
    patch_dir: Union[str, Path],
    all_da_flag: int = 0,
    no_weed_adjacent: Optional[int] = None,
    no_weed_noisy: Optional[int] = None,
    psver: int = 1,
) -> None:
    """Module-level entry point (mirrors MATLAB ``ps_weed``)."""
    weeder = PSWeeder(patch_dir=patch_dir, psver=psver)
    weeder.run(
        all_da_flag=all_da_flag,
        no_weed_adjacent=no_weed_adjacent,
        no_weed_noisy=no_weed_noisy,
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 4: Weed out neighbouring/noisy PS (ps_weed)"
    )
    parser.add_argument(
        "patch_dir", type=str,
        help="Path to patch directory",
    )
    parser.add_argument(
        "--no-weed-adjacent", type=int, default=None,
        help="1=skip adjacent weeding (default: determined by config)",
    )
    parser.add_argument(
        "--no-weed-noisy", type=int, default=None,
        help="1=skip noise weeding (default: determined by config)",
    )
    parser.add_argument(
        "--psver", type=int, default=1,
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    ps_weed(
        args.patch_dir,
        no_weed_adjacent=args.no_weed_adjacent,
        no_weed_noisy=args.no_weed_noisy,
        psver=args.psver,
    )
