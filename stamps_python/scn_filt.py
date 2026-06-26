#!/usr/bin/env python3
"""
scn_filt.py — StaMPS Step 8: Spatially-correlated noise (SCN) estimation
========================================================================

Estimates spatially-correlated noise in unwrapped phase by:
  1. Subtracting SCLA (from Step 7) from unwrapped phase
  2. Optional deramp of selected IFGs (orbit ramps)
  3. Low-pass filter in time (Gaussian window) on edge differences
  4. Solve for high-frequency pixel phase from edge differences
  5. Low-pass filter in space (Gaussian weights) to get SCN

Data IO (HDF5 only):
- Read: ps{v}.h5, phuw{v}.h5 or phuw_sb{v}.h5 (SB), scla{v}.h5 from patch_dir.
- Write: scn{v}.h5 (ph_scn_slave, ph_hpt, ph_ramp).

MATLAB correspondence
---------------------
+---------------------------------------+------------------------------------------+
| MATLAB                                | Python                                    |
+=======================================+==========================================+
| ps_scn_filt()                         | ScnFiltPipeline(patch_dir, psver).run()   |
| load(psname), load(phuwname), scla    | HDF5 only: ps{v}.h5, phuw*{v}.h5, scla{v}.h5 |
| triangle → scnfilt.2.edge             | scipy.spatial.Delaunay → edges            |
| Gaussian time weights (time_win)     | exp(-dt^2/2/time_win^2), master excluded  |
| A\\dph_hpt (edge incidence)           | scipy.sparse linalg.spsolve               |
| Gaussian spatial weights (scn_wavelength) | exp(-dist_sq/(2*scn_wavelength^2))  |
| save(scnname, ph_scn_slave, ...)     | scn{v}.h5                                 |
+---------------------------------------+------------------------------------------+

Filter parameters (aligned with MATLAB ps_scn_filt.m):
- Time: weight = exp(-(day(i1)-day)^2 / (2*time_win^2)); master weight = 0; then normalize.
- Space: sigma_sq_times_2 = 2*scn_wavelength^2; weight = exp(-dist_sq/sigma_sq_times_2); normalize.

Refactored from: ps_scn_filt.m (Andy Hooper 2006+, DB 2011)
"""

from __future__ import annotations

import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy import sparse

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
# Helpers: load HDF5 / parse config
# ===========================================================================

def _load_h5(path: Path, keys: List[str]) -> Dict[str, np.ndarray]:
    """Load requested keys from HDF5; missing keys are skipped."""
    data: Dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as hf:
        for k in keys:
            if k in hf:
                data[k] = np.asarray(hf[k][:])
    return data


def _load_h5_required(patch_dir: Path, basename: str,
                      keys: List[str]) -> Dict[str, np.ndarray]:
    """Load keys from patch_dir/basename.h5 only. Raises FileNotFoundError if missing.

    Step 8 uses only .h5 files produced by previous steps (no .mat for computation).
    """
    h5_path = patch_dir / f"{basename}.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Step 8 requires {h5_path.name} (from previous steps) in {patch_dir}"
        )
    return _load_h5(h5_path, keys)


def _parse_drop_index(raw) -> np.ndarray:
    """Return 0-based int array of IFG indices to drop."""
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
    if arr.size > 0 and arr.min() >= 1:
        arr = arr - 1
    return arr[arr >= 0]


def _parse_deramp_ifg(raw, unwrap_ifg_index: np.ndarray) -> np.ndarray:
    """Return deramp_ix: 0-based column indices into the unwrap IFG list.

    MATLAB: deramp_ifg='all' -> 1:ps.n_ifg; else use list; then intersect with unwrap_ifg_index.
    deramp_ix(i) = find(unwrap_ifg_index == deramp_ifg(i)).
    """
    if raw is None:
        return np.array([], dtype=np.int32)
    if isinstance(raw, str) and raw.strip().lower() == "all":
        return np.arange(unwrap_ifg_index.size, dtype=np.int32)
    # List of 1-based IFG indices
    if isinstance(raw, (bytes, str)):
        s = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        s = s.strip().strip("[]")
        if not s:
            return np.array([], dtype=np.int32)
        try:
            vals = [int(float(x)) for x in s.replace(",", " ").split()]
        except ValueError:
            return np.array([], dtype=np.int32)
        deramp_ifg_1based = np.array(vals, dtype=np.int32)
    else:
        deramp_ifg_1based = np.atleast_1d(np.asarray(raw).ravel()).astype(np.int32)
    if deramp_ifg_1based.size == 0 or np.all(deramp_ifg_1based == 0):
        return np.array([], dtype=np.int32)
    # unwrap_ifg_index is 0-based in our Python; MATLAB is 1-based
    unwrap_1based = unwrap_ifg_index + 1
    deramp_ix = np.array([
        np.searchsorted(unwrap_1based, d) for d in deramp_ifg_1based
        if np.any(unwrap_1based == d)
    ], dtype=np.int32)
    return deramp_ix


# ===========================================================================
# Build Delaunay edges (replace triangle executable)
# ===========================================================================

def _build_edges_delaunay(xy: np.ndarray) -> np.ndarray:
    """Build edge list from Delaunay triangulation of xy (n_ps, 3) with cols 1,2 = x,y.

    Returns edges (n_edges, 2) with 0-based node indices; each row is (node_i, node_j).
    """
    x = np.asarray(xy[:, 1], dtype=np.float64)
    y = np.asarray(xy[:, 2], dtype=np.float64)
    tri = Delaunay(np.column_stack([x, y]))
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(simplex[i]), int(simplex[j])
                edges_set.add((min(a, b), max(a, b)))
    return np.array(list(edges_set), dtype=np.int32)


# ===========================================================================
# Main pipeline
# ===========================================================================

class ScnFiltPipeline:
    """Step 8: Estimate spatially-correlated noise (SCN) in unwrapped phase.

    MATLAB: ps_scn_filt() when scn_kriging_flag ~= 'y'.
    """

    def __init__(self, patch_dir: Path, psver: int = 2):
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver
        self.cfg = StampsConfig(work_dir=self.patch_dir)
        if not self.cfg._loaded:
            self.cfg.load()

    def run(self) -> None:
        use_kriging = str(self.cfg.getparm("scn_kriging_flag") or "n").strip().lower()
        if use_kriging == "y":
            logger.warning(
                "scn_kriging_flag='y' is not implemented; using Gaussian filter (ps_scn_filt)."
            )
        self._run_gaussian()

    def _run_gaussian(self) -> None:
        """Gaussian time + space low-pass (ps_scn_filt.m logic)."""
        logger.info("Estimating other spatially-correlated noise...")
        v = self.psver
        cfg = self.cfg

        # Parameters (aligned with MATLAB getparm)
        time_win = float(cfg.getparm("scn_time_win") or 365.0)
        scn_wavelength = float(cfg.getparm("scn_wavelength") or 100.0)
        deramp_ifg_raw = cfg.getparm("scn_deramp_ifg")
        drop_ifg_index = _parse_drop_index(cfg.getparm("drop_ifg_index"))
        small_baseline_flag = str(cfg.getparm("small_baseline_flag") or "n").strip().lower()

        logger.info(
            "   SCN filter parameters: time_win=%.1f days, scn_wavelength=%.1f m (Gaussian)",
            time_win, scn_wavelength,
        )

        # Load data (HDF5 only)
        ps = self._load_ps(v)
        use_sb_ifgs = small_baseline_flag == "y" and not (self.patch_dir / f"phuw{v}.h5").is_file()
        uw = self._load_phuw(v, use_sb_ifgs)
        n_ps = ps["n_ps"]
        n_ifg_full = int(np.asarray(ps["n_ifg"]).ravel()[0])
        day_full = np.asarray(ps["day"]).ravel()
        master_day = float(np.asarray(ps["master_day"]).ravel()[0])

        # Unwrap IFG index (0-based). MATLAB: SB -> [1:ps.n_image]; else setdiff([1:n_ifg], drop)
        if small_baseline_flag == "y" and not use_sb_ifgs:
            n_image = ps.get("n_image")
            if n_image is None:
                n_image = uw["ph_uw"].shape[1]
            unwrap_ifg_index = np.arange(n_image, dtype=np.int32)
            n_ifg_full = n_image
        elif small_baseline_flag == "y":
            n_image = ps.get("n_image")
            if n_image is None:
                n_image = uw["ph_uw"].shape[1]
            unwrap_ifg_index = np.arange(n_image, dtype=np.int32)
        else:
            unwrap_ifg_index = np.setdiff1d(
                np.arange(n_ifg_full, dtype=np.int32),
                drop_ifg_index,
            )
        n_ifg = unwrap_ifg_index.size
        # day(i) = ps.day(unwrap_ifg_index(i)) in MATLAB (1-based there)
        day = day_full[unwrap_ifg_index].astype(np.float64)

        # Master index in unwrap list (0-based) for time-weight zeroing. MATLAB: master_ix=sum(ps.master_day>ps.day)+1
        master_ix_0 = 0
        for i in range(day.size):
            if day[i] == master_day:
                master_ix_0 = i
                break

        # ph_all: (n_ps, n_ifg), MATLAB: ph_all = uw.ph_uw(:, unwrap_ifg_index)
        ph_all = np.asarray(uw["ph_uw"][:, unwrap_ifg_index], dtype=np.float64)
        scla_path = self.patch_dir / f"scla{v}.h5"
        if use_sb_ifgs:
            scla_path = self.patch_dir / f"scla_sb{v}.h5"
        if scla_path.is_file():
            scla = _load_h5(scla_path, ["ph_scla", "C_ps_uw", "ph_ramp"])
            ph_scla = np.asarray(scla["ph_scla"])
            C_ps_uw = np.asarray(scla["C_ps_uw"]).ravel()
            if ph_scla.shape == ph_all.shape:
                ph_all = ph_all - ph_scla
            elif ph_scla.ndim == 2 and ph_scla.shape[1] > int(unwrap_ifg_index.max()):
                ph_all = ph_all - ph_scla[:, unwrap_ifg_index]
            subtract_c_for_scn = not (small_baseline_flag == "y" and not use_sb_ifgs)
            if C_ps_uw.size == n_ps and subtract_c_for_scn:
                ph_all = ph_all - np.broadcast_to(C_ps_uw[:, np.newaxis], ph_all.shape)
            elif C_ps_uw.size == n_ps:
                logger.info(
                    "   SB single-master SCN: leaving C_ps_uw out of the temporal filter"
                )
            if "ph_ramp" in scla and scla["ph_ramp"] is not None:
                ph_ramp_scla = np.asarray(scla["ph_ramp"])
                if ph_ramp_scla.shape == ph_all.shape:
                    ph_all = ph_all - ph_ramp_scla
                elif ph_ramp_scla.ndim == 2 and ph_ramp_scla.shape[1] > int(unwrap_ifg_index.max()):
                    ph_all = ph_all - ph_ramp_scla[:, unwrap_ifg_index]
        else:
            logger.warning("   SCLA file not found for SCN correction: %s", scla_path.name)
        ph_all = np.nan_to_num(ph_all, nan=0.0)

        logger.info("   Number of points per ifg: %d", n_ps)

        # Phase std before filtering (for logging)
        std_before = float(np.nanstd(ph_all))
        logger.info("   Phase std before SCN filter: %.6f rad", std_before)

        # Build edges (Delaunay)
        xy = np.asarray(ps["xy"])
        if xy.shape[1] < 3:
            xy = np.column_stack([np.arange(n_ps, dtype=np.float64), xy[:, 0], xy[:, 1]])
        edges = _build_edges_delaunay(xy)
        n_edges = edges.shape[0]
        logger.info("   Number of edges: %d", n_edges)

        # Optional deramp
        deramp_ix = _parse_deramp_ifg(deramp_ifg_raw, unwrap_ifg_index)
        ph_ramp = np.zeros((n_ps, deramp_ix.size), dtype=np.float64)
        if deramp_ix.size > 0:
            logger.info("   Deramping %d selected IFGs...", deramp_ix.size)
            G = np.column_stack([
                np.ones(n_ps),
                xy[:, 1],
                xy[:, 2]
            ]).astype(np.float64)
            for j, col in enumerate(deramp_ix):
                d = ph_all[:, col]
                m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
                ph_this_ramp = G @ m
                ph_all[:, col] = ph_all[:, col] - ph_this_ramp
                ph_ramp[:, j] = ph_this_ramp

        # Edge differences: dph(e,t) = ph_all(node2,t) - ph_all(node1,t)
        node1 = edges[:, 0]
        node2 = edges[:, 1]
        dph = ph_all[node2, :] - ph_all[node1, :]  # (n_edges, n_ifg)

        # Time low-pass: Gaussian window, exclude master
        # weight_factor(i1) = exp(-(day(i1)-day)^2/2/time_win^2); weight(master_ix)=0; normalize
        dph_lpt = np.zeros_like(dph)
        logger.info("   Low-pass filtering pixel-pairs in time (time_win=%.1f days)...", time_win)
        for i1 in range(n_ifg):
            time_diff_sq = (day[i1] - day) ** 2
            weight = np.exp(-time_diff_sq / (2.0 * time_win ** 2))
            weight[master_ix_0] = 0.0
            wsum = weight.sum()
            if wsum > 0:
                weight = weight / wsum
            dph_lpt[:, i1] = dph @ weight

        dph_hpt = dph - dph_lpt

        # Solve A @ ph_hpt = dph_hpt (ref_ix=0, so node 0 is fixed at 0)
        # Incidence: row e has -1 at node1(e), +1 at node2(e). Build sparse, drop column 0.
        ref_ix = 0
        logger.info("   Solving for high-frequency (in time) pixel phase...")
        row = np.repeat(np.arange(n_edges), 2)
        col = np.column_stack([node1, node2]).ravel()
        data = np.tile([-1.0, 1.0], n_edges)
        A_full_sp = sparse.csr_matrix((data, (row, col)), shape=(n_edges, n_ps))
        A_sp = A_full_sp[:, 1:]  # drop column 0 (ref)
        # Least-squares: (A'A)x = A'b; factorize once, solve per IFG
        AtA = A_sp.T @ A_sp
        Atb = A_sp.T @ dph_hpt
        solve_lu = sparse.linalg.factorized(AtA)
        ph_hpt_reduced = np.column_stack([solve_lu(Atb[:, i]) for i in range(n_ifg)])
        ph_hpt = np.vstack([
            ph_hpt_reduced[:ref_ix],
            np.zeros((1, n_ifg)),
            ph_hpt_reduced[ref_ix:]
        ])

        # Add ramp back to deramped IFGs
        for j, col in enumerate(deramp_ix):
            if col < n_ifg:
                ph_hpt[:, col] += ph_ramp[:, j]

        ph_hpt = ph_hpt.astype(np.float32)

        # Spatial low-pass: Gaussian weights
        # sigma_sq_times_2 = 2*scn_wavelength^2; patch_dist = scn_wavelength*4
        sigma_sq_times_2 = 2.0 * scn_wavelength ** 2
        patch_dist = scn_wavelength * 4.0
        patch_dist_sq = patch_dist ** 2
        x_coord = np.asarray(xy[:, 1], dtype=np.float64)
        y_coord = np.asarray(xy[:, 2], dtype=np.float64)

        ph_scn = np.full((n_ps, n_ifg), np.nan, dtype=np.float64)
        logger.info("   Low-pass filtering in space (scn_wavelength=%.1f m)...", scn_wavelength)

        # Use cKDTree for fast range queries (patch_dist)
        tree = cKDTree(np.column_stack([x_coord, y_coord]))
        for i in range(n_ps):
            nbr_ix = tree.query_ball_point([x_coord[i], y_coord[i]], r=patch_dist)
            nbr_ix = np.asarray(nbr_ix, dtype=np.int32)
            if nbr_ix.size == 0:
                nbr_ix = np.array([i])
            dist_sq = (x_coord[nbr_ix] - x_coord[i]) ** 2 + (y_coord[nbr_ix] - y_coord[i]) ** 2
            weight = np.exp(-dist_sq / sigma_sq_times_2)
            weight = weight / weight.sum()
            ph_scn[i, :] = weight @ ph_hpt[nbr_ix, :]
            if (i + 1) % 10000 == 0:
                logger.info("   %d PS processed", i + 1)

        # Re-ref to first PS
        ph_scn = ph_scn - ph_scn[0:1, :]

        # Phase std after filtering (residual = high-pass - SCN)
        ph_noise_res = ph_hpt - ph_scn
        std_after = np.nanstd(ph_noise_res)
        logger.info("   Phase std after SCN filter (residual): %.6f rad", std_after)
        logger.info("   Filter: time_win=%.1f days, scn_wavelength=%.1f m (Gaussian).",
                    time_win, scn_wavelength)

        # ph_scn_slave: full size like ph_uw; fill unwrap columns, master column 0 (MATLAB: ph_scn_slave(:,master_ix)=0)
        ph_scn_slave = np.zeros((n_ps, n_ifg_full), dtype=np.float32)
        ph_scn_slave[:, unwrap_ifg_index] = ph_scn.astype(np.float32)
        master_col = unwrap_ifg_index[master_ix_0]
        ph_scn_slave[:, master_col] = 0.0

        # Save
        scn_path = self.patch_dir / f"scn{v}.h5"
        with h5py.File(str(scn_path), "w") as hf:
            hf.create_dataset("ph_scn_slave", data=ph_scn_slave, compression="gzip")
            hf.create_dataset("ph_hpt", data=ph_hpt, compression="gzip")
            hf.create_dataset("ph_ramp", data=ph_ramp.astype(np.float32), compression="gzip")
        logger.info("   Saved %s (ph_scn_slave %s, ph_hpt %s)",
                    scn_path.name, ph_scn_slave.shape, ph_hpt.shape)

    def _load_ps(self, v: int) -> Dict[str, Any]:
        """Load PS metadata from ps{v}.h5 only (HDF5)."""
        keys = ["n_ps", "n_ifg", "n_image", "master_ix", "master_day", "xy", "day"]
        raw = _load_h5_required(self.patch_dir, f"ps{v}", keys)
        for k in ["n_ps", "n_ifg", "master_ix"]:
            if k in raw:
                raw[k] = int(np.asarray(raw[k]).ravel()[0])
        if "n_image" in raw:
            raw["n_image"] = int(np.asarray(raw["n_image"]).ravel()[0])
        if "master_day" in raw:
            raw["master_day"] = np.asarray(raw["master_day"]).ravel()
        if "day" in raw:
            raw["day"] = np.asarray(raw["day"]).ravel()
        if "xy" in raw:
            raw["xy"] = np.asarray(raw["xy"])
            if raw["xy"].ndim == 1:
                raw["xy"] = raw["xy"].reshape(-1, 3)
        return raw

    def _load_phuw(self, v: int, small_baseline: bool) -> Dict[str, np.ndarray]:
        """Load unwrapped phase from phuw_sb{v}.h5 (SB) or phuw{v}.h5 (PS). HDF5 only."""
        basename = f"phuw_sb{v}" if small_baseline else f"phuw{v}"
        return _load_h5_required(self.patch_dir, basename, ["ph_uw", "msd"])


def ps_scn_filt(patch_dir: Optional[Path] = None, psver: int = 2) -> None:
    """MATLAB-style entry point for Step 8."""
    ScnFiltPipeline(patch_dir or Path.cwd(), psver=psver).run()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run StaMPS Step 8 SCN filtering.")
    parser.add_argument("patch_dir", nargs="?", default=".", help="Project or patch directory")
    parser.add_argument("--psver", type=int, default=2, help="PS version number")
    args = parser.parse_args()
    ps_scn_filt(Path(args.patch_dir), psver=args.psver)


if __name__ == "__main__":
    _main()
