#!/usr/bin/env python3
"""
ps_selection_final.py — Python port of ps_select.m  (StaMPS Step 3)
====================================================================

Select PS pixels based on gamma (coherence) and D_A (dispersion of
amplitude).

MATLAB → Python mapping
-----------------------
+--------------------------------------+--------------------------------------------------+
| MATLAB concept                       | Python mapping                                   |
+======================================+==================================================+
| ``ps_select(reest_flag, plot_flag)`` | ``PSSelector(patch_dir).run(reest_flag)``        |
+--------------------------------------+--------------------------------------------------+
| D_A binning + histogram thresholding | ``_compute_coh_threshold()`` (vectorised)        |
+--------------------------------------+--------------------------------------------------+
| ``clap_filt_patch(ph, α, β, lp)``   | ``clap_filt_patch()`` (single-window FFT filter) |
+--------------------------------------+--------------------------------------------------+
| Per-pixel re-estimation loop         | ``_reestimate_gamma()`` (vectorised topofit)     |
+--------------------------------------+--------------------------------------------------+
| ``polyfit / polyval`` with centering | ``np.polynomial.polynomial.polyfit`` or manual   |
+--------------------------------------+--------------------------------------------------+
| ``stamps_save('select1', …)``        | ``_save_select_h5(…, select1.h5)``               |
+--------------------------------------+--------------------------------------------------+

Numeric type conventions
------------------------
* ``coh_ps``, ``K_ps``, ``C_ps`` are ``float64`` (MATLAB ``double``).
* ``ph_patch2``, ``ph_res2`` are ``complex64`` / ``float32`` (MATLAB ``single``).
* ``ix`` indices in the output HDF5 are **1-based** (MATLAB convention)
  to maintain compatibility.
* ``ifg_index`` in the output HDF5 is also **1-based**.
* ``keep_ix`` is a boolean array stored as uint8 (0/1).

Performance notes
-----------------
* **No explicit pixel loops** for the initial threshold computation
  (D_A binning, histograms, cumulative sums are all vectorised).
* The re-estimation loop over selected pixels *does* iterate per-pixel
  for the CLAP filter patch extraction (as in MATLAB), but the heavy
  ``ps_topofit`` call is **batch-vectorised** for all selected pixels
  at once.
* Large Step-3 diagnostic/reuse arrays are skipped by default to avoid very
  slow HDF5 writes.  Set ``STAMPS_SAVE_SELECT_PH_RES2=1`` and/or
  ``STAMPS_SAVE_SELECT_PH_PATCH2=1`` to keep them for validation or
  ``reest_flag=2`` reuse.

Refactored from: ps_select.m (Andy Hooper, June 2006)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian as _scipy_gaussian

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig, load_mat  # noqa: E402
from gamma_est import ps_topofit_batch, _gausswin  # noqa: E402

logger = logging.getLogger("stamps")
_FAST_COMPRESSION = {}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
            # v7.3 or string/bytes: try numeric conversion, else empty
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
# clap_filt_patch — port of clap_filt_patch.m
# ============================================================================

def clap_filt_patch(
    ph: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.3,
    low_pass: Optional[np.ndarray] = None,
) -> np.ndarray:
    """CLAP filter for a *single* patch tile (no overlap-add).

    Port of ``clap_filt_patch.m`` (Andy Hooper, June 2006).

    Unlike ``clap_filt`` (full-image overlap-add), this operates on a
    single window-sized tile:

    1. FFT the tile.
    2. Smooth the spectral magnitude with a 7×7 Gaussian kernel.
    3. Normalise by median, raise to power *alpha*, subtract 1, clamp negatives.
    4. Combine with low-pass:  ``G = H * beta + low_pass``.
    5. Inverse-FFT.

    Parameters
    ----------
    ph : ndarray (n_win, n_win), complex
        Single tile of phase grid.
    alpha, beta : float
        CLAP parameters.
    low_pass : ndarray or None
        Butterworth low-pass (same shape as *ph*), fftshift-ed.

    Returns
    -------
    ph_out : ndarray (n_win, n_win), complex
    """
    ph = np.copy(ph)
    ph[np.isnan(ph)] = 0

    if low_pass is None:
        low_pass = np.zeros_like(ph, dtype=np.float64)

    gw7 = _gausswin(7)
    B = np.outer(gw7, gw7)

    ph_fft = np.fft.fft2(ph)
    H = np.abs(ph_fft)
    # MATLAB: H = ifftshift(filter2(B, fftshift(H)))
    H_shifted = np.fft.fftshift(H)
    H_smooth = fftconvolve(H_shifted, B, mode="same")
    H = np.fft.ifftshift(H_smooth)

    median_H = np.median(H)
    if median_H != 0:
        H = H / median_H

    H = H ** alpha
    H = H - 1.0
    H[H < 0] = 0.0

    G = H * beta + low_pass
    ph_out = np.fft.ifft2(ph_fft * G)
    return ph_out


def _smooth_spectral_magnitude(
    H: np.ndarray,
    kernel_1d: np.ndarray,
    axes: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Apply MATLAB ``filter2(B, fftshift(H))`` to all IFG slices.

    ``B`` is the outer product of the symmetric 1-D Gaussian kernel, so this
    uses two spatial 1-D convolutions instead of looping over IFGs with a
    separate 2-D convolution.  ``mode='constant'`` matches the zero-padding
    behaviour of MATLAB ``filter2(..., 'same')`` closely.
    """
    H_shifted = np.fft.fftshift(H, axes=axes)
    H_shifted = convolve1d(
        H_shifted,
        weights=kernel_1d,
        axis=axes[0],
        mode="constant",
        cval=0.0,
    )
    H_shifted = convolve1d(
        H_shifted,
        weights=kernel_1d,
        axis=axes[1],
        mode="constant",
        cval=0.0,
    )
    return np.fft.ifftshift(H_shifted, axes=axes)


# ============================================================================
# Threshold computation helpers
# ============================================================================

def _compute_coh_threshold_single_bin(
    coh_chunk: np.ndarray,
    coh_bins: np.ndarray,
    Nr_dist: np.ndarray,
    low_coh_thresh: int,
    select_method: str,
    max_percent_rand: float,
) -> float:
    """Compute the coherence threshold for one D_A bin.

    MATLAB correspondence (ps_select.m lines 143-179)::

        coh_chunk = pm.coh_ps(D_A > D_A_max(i) & D_A <= D_A_max(i+1));
        coh_chunk = coh_chunk(coh_chunk ~= 0);
        Na = hist(coh_chunk, pm.coh_bins);
        Nr = Nr_dist * sum(Na(1:low_coh_thresh)) / sum(Nr_dist(1:low_coh_thresh));
        ...
        percent_rand = fliplr(cumsum(fliplr(Nr)) ./ cumsum(fliplr(Na)) * 100);
        ok_ix = find(percent_rand < max_percent_rand);
        ... polyfit / polyval ...

    Returns
    -------
    min_coh : float
        Coherence threshold for this bin. NaN if indeterminate.
    """
    coh_chunk = coh_chunk[coh_chunk != 0]
    if len(coh_chunk) == 0:
        return np.nan

    # Histogram of actual coherence — MATLAB: Na = hist(coh_chunk, coh_bins)
    Na, _ = np.histogram(coh_chunk, bins=np.arange(0.0, 1.01, 0.01))
    Na = Na.astype(np.float64)

    # Scale random distribution
    sum_Na_low = Na[:low_coh_thresh].sum()
    sum_Nr_low = Nr_dist[:low_coh_thresh].sum()
    if sum_Nr_low == 0:
        return np.nan
    Nr = Nr_dist * (sum_Na_low / sum_Nr_low)

    Na_safe = Na.copy()
    Na_safe[Na_safe == 0] = 1.0

    # Cumulative random ratio — MATLAB lines 158-162
    if select_method.upper() == "PERCENT":
        # fliplr(cumsum(fliplr(Nr)) ./ cumsum(fliplr(Na)) * 100)
        cum_Nr = np.cumsum(Nr[::-1])[::-1]
        cum_Na = np.cumsum(Na_safe[::-1])[::-1]
        cum_Na[cum_Na == 0] = 1.0
        percent_rand = cum_Nr / cum_Na * 100.0
    else:
        # DENSITY mode: absolute count
        percent_rand = np.cumsum(Nr[::-1])[::-1]

    # Find first bin below max_percent_rand — MATLAB line 163
    ok_ix = np.where(percent_rand < max_percent_rand)[0]
    if len(ok_ix) == 0:
        return 1.0  # no threshold meets criteria

    min_ok = ok_ix.min()
    min_fit_ix = min_ok - 3
    if min_fit_ix <= 0:
        return np.nan

    max_fit_ix = min(min_ok + 2, 99)  # max index 99 (100 bins, 0-based)

    # Polynomial fit with centering — MATLAB: [p, S, mu] = polyfit(x, y, 3)
    # x = percent_rand(min_fit_ix:max_fit_ix), y = [min_fit_ix*0.01 : 0.01 : max_fit_ix*0.01]
    x_fit = percent_rand[min_fit_ix: max_fit_ix + 1]
    # MATLAB indexing is 1-based, so y = [min_fit_ix*0.01, ..., max_fit_ix*0.01]
    # In MATLAB min_fit_ix is 1-based; here it's 0-based, but the bin centres
    # are at (index+1)*0.01 for MATLAB.  Actually MATLAB line 174:
    #   y = [min_fit_ix*0.01:0.01:max_fit_ix*0.01]
    # where min_fit_ix is 1-based → maps to our (min_fit_ix+1)*0.01
    y_fit = np.arange(min_fit_ix + 1, max_fit_ix + 2) * 0.01

    if len(x_fit) < 2:
        return np.nan

    # MATLAB polyfit with centering: [p, S, mu] = polyfit(x, y, 3)
    # mu = [mean(x), std(x)];  x_scaled = (x - mu(1)) / mu(2)
    mu_x = x_fit.mean()
    std_x = x_fit.std(ddof=0)  # MATLAB std in polyfit uses N (not N-1)
    if std_x == 0:
        return np.nan
    x_scaled = (x_fit - mu_x) / std_x

    deg = min(3, len(x_fit) - 1)
    coeffs = np.polyfit(x_scaled, y_fit, deg)

    # Evaluate at max_percent_rand — MATLAB: polyval(p, max_percent_rand, [], mu)
    x_eval = (max_percent_rand - mu_x) / std_x
    min_coh = np.polyval(coeffs, x_eval)
    return float(min_coh)


# ============================================================================
# HDF5 persistence for select1
# ============================================================================

def _save_select_h5(
    h5_path: Path,
    *,
    ix: np.ndarray,
    keep_ix: np.ndarray,
    ph_patch2: np.ndarray,
    ph_res2: np.ndarray,
    K_ps2: np.ndarray,
    C_ps2: np.ndarray,
    coh_ps2: np.ndarray,
    coh_thresh: np.ndarray,
    coh_thresh_coeffs: Optional[np.ndarray],
    clap_alpha: float,
    clap_beta: float,
    n_win: int,
    max_percent_rand: float,
    gamma_stdev_reject: float,
    small_baseline_flag: str,
    ifg_index: np.ndarray,
    save_ph_patch2: bool = False,
    save_ph_res2: bool = False,
) -> None:
    """Persist Step-3 outputs to ``select1.h5``.

    Dataset layout mirrors MATLAB ``stamps_save(selectname, …)``
    (ps_select.m line 463).

    For MATLAB compatibility:
      * ``ix`` is 1-based (matching MATLAB).
      * ``ifg_index`` is 1-based (matching MATLAB).
      * ``keep_ix`` is stored as uint8 (0/1).
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing Step-3 HDF5 output: %s", h5_path)

    logger.info(
        "Step-3 save options: ph_patch2=%s, ph_res2=%s",
        "yes" if save_ph_patch2 else "no",
        "yes" if save_ph_res2 else "no",
    )

    with h5py.File(str(h5_path), "w") as hf:
        # Large diagnostic/reuse arrays are optional because HDF5 writes are
        # disproportionately expensive on large patches.  Step 4 can continue
        # without ph_res2; reest_flag=2 reuse still requires ph_patch2.
        if save_ph_patch2:
            hf.create_dataset("ph_patch2", data=np.ascontiguousarray(ph_patch2, dtype=np.complex64), **_FAST_COMPRESSION)
        if save_ph_res2:
            hf.create_dataset("ph_res2", data=np.ascontiguousarray(ph_res2, dtype=np.float32), **_FAST_COMPRESSION)
        hf.attrs["ph_patch2_saved"] = bool(save_ph_patch2)
        hf.attrs["ph_res2_saved"] = bool(save_ph_res2)

        # Per-pixel vectors
        hf.create_dataset("ix", data=ix.astype(np.int32))
        hf.create_dataset("keep_ix", data=keep_ix.astype(np.uint8))
        hf.create_dataset("K_ps2", data=K_ps2.astype(np.float64))
        hf.create_dataset("C_ps2", data=C_ps2.astype(np.float64))
        hf.create_dataset("coh_ps2", data=coh_ps2.astype(np.float64))
        hf.create_dataset("coh_thresh", data=coh_thresh.astype(np.float64))

        # Threshold coefficients (may be empty)
        if coh_thresh_coeffs is not None and len(coh_thresh_coeffs) > 0:
            hf.create_dataset("coh_thresh_coeffs", data=np.asarray(coh_thresh_coeffs, dtype=np.float64))
        else:
            hf.create_dataset("coh_thresh_coeffs", data=np.array([], dtype=np.float64))

        # Scalars / small arrays
        hf.create_dataset("clap_alpha", data=np.float64(clap_alpha))
        hf.create_dataset("clap_beta", data=np.float64(clap_beta))
        hf.create_dataset("n_win", data=np.int32(n_win))
        hf.create_dataset("max_percent_rand", data=np.float64(max_percent_rand))
        hf.create_dataset("gamma_stdev_reject", data=np.float64(gamma_stdev_reject))

        # String stored as fixed-length bytes for h5py compatibility
        hf.create_dataset("small_baseline_flag", data=small_baseline_flag)
        hf.create_dataset("ifg_index", data=ifg_index.astype(np.int32))

    sz = h5_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", h5_path.name, sz)


# ============================================================================
# Main selector class
# ============================================================================

class PSSelector:
    """Select PS pixels from Step-2 gamma estimates — port of ``ps_select.m``.

    Parameters
    ----------
    patch_dir : Path
        Directory containing ``ps1.h5``, ``pm1.h5``, and auxiliary files.
    psver : int
        PS version number (default 1).
    """

    def __init__(self, patch_dir: Union[str, Path], psver: int = 1) -> None:
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver

        # Config singleton
        self._cfg = StampsConfig(work_dir=self.patch_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, reest_flag: int = 0) -> None:
        """Execute PS selection.

        Parameters
        ----------
        reest_flag : int
            0 = re-estimate gamma for initially selected PS (default).
            1 = skip re-estimation completely.
            2 = reuse previously calculated re-estimation.
            3 = re-estimate for all candidate pixels.

        MATLAB correspondence
        ---------------------
        ``ps_select(reest_flag, plot_flag)`` in ps_select.m.
        """
        t_start = time.time()
        logger.info("Selecting stable-phase pixels...")

        # --- Load parameters ---
        parms = self._load_parameters()
        cfg = self._cfg

        small_baseline_flag = parms["small_baseline_flag"]
        if small_baseline_flag == "y":
            low_coh_thresh = 15
        else:
            low_coh_thresh = 31

        # --- Load Step-1 data (ps1.h5 / ps1.mat) ---
        (ph, bperp_1d, n_ifg, n_ps, xy, master_ix,
         ifgday_ix, D_A) = self._load_ps_data(parms)

        # --- Load Step-2 data (pm1.h5 / pm1.mat) ---
        (coh_ps, coh_bins, Nr_dist, K_ps_orig, C_ps_orig,
         ph_res_orig, ph_patch_orig, ph_grid, ph_weight,
         n_trial_wraps, grid_ij, low_pass) = self._load_pm_data()

        # --- Handle PS mode: remove master column ---
        # MATLAB (ps_select.m lines 93-101):
        #   if ~strcmpi(small_baseline_flag, 'y')
        #       no_master_ix = setdiff(1:n_ifg, master_ix);
        #       ph = ph(:, no_master_ix);
        #       bperp = bperp(no_master_ix);
        #       n_ifg = length(no_master_ix);
        #   end
        drop_ifg_index = parms["drop_ifg_index"]

        if small_baseline_flag != "y":
            midx_0 = int(master_ix) - 1  # 0-based
            keep_cols = np.concatenate([np.arange(midx_0), np.arange(midx_0 + 1, ph.shape[1])])
            # ifg_index for dropping — MATLAB line 96:
            #   ifg_index = setdiff(ifg_index, ps.master_ix)
            #   ifg_index(ifg_index > master_ix) -= 1
            all_ifg_0 = np.arange(ph.shape[1])  # 0-based, original n_ifg
            # Remove master from all_ifg and drop_ifg_index
            ifg_index_1based = np.setdiff1d(np.arange(1, ph.shape[1] + 1), [int(master_ix)])
            # Remove dropped ifgs
            if len(drop_ifg_index) > 0:
                ifg_index_1based = np.setdiff1d(ifg_index_1based, drop_ifg_index)
            # Adjust: ifg_index entries > master_ix get decremented by 1
            ifg_index_1based_adj = ifg_index_1based.copy()
            ifg_index_1based_adj[ifg_index_1based_adj > int(master_ix)] -= 1

            ph = ph[:, keep_cols]
            bperp_1d = bperp_1d[keep_cols]
            n_ifg = len(keep_cols)
            ifg_index_0 = ifg_index_1based_adj.astype(np.int64) - 1  # 0-based
        else:
            # SB mode: ifg_index = setdiff(1:n_ifg, drop_ifg_index)
            all_ifg_1 = np.arange(1, n_ifg + 1)
            if len(drop_ifg_index) > 0:
                ifg_index_1based = np.setdiff1d(all_ifg_1, drop_ifg_index)
            else:
                ifg_index_1based = all_ifg_1.copy()
            ifg_index_1based_adj = ifg_index_1based
            ifg_index_0 = ifg_index_1based.astype(np.int64) - 1

        # For saving: 1-based ifg_index
        ifg_index_save = ifg_index_1based_adj.astype(np.int32)

        # --- D_A binning (lines 115-128) ---
        D_A_bins, D_A_all = self._compute_da_bins(D_A, coh_ps)

        # --- Compute max_percent_rand for DENSITY mode (lines 130-133) ---
        select_method = parms["select_method"]
        if select_method.upper() == "PERCENT":
            max_percent_rand = float(parms["percent_rand"])
        else:
            # MATLAB: patch_area = prod(max(xy(:,2:3)) - min(xy(:,2:3))) / 1e6
            xy_range = xy[:, 1:3].max(axis=0) - xy[:, 1:3].min(axis=0)
            patch_area = float(np.prod(xy_range)) / 1e6
            max_percent_rand = float(parms["density_rand"]) * patch_area / (len(D_A_bins) - 1)
            logger.info("DENSITY mode: patch_area=%.4f km², max_percent_rand=%.4f",
                        patch_area, max_percent_rand)

        gamma_stdev_reject = parms["gamma_stdev_reject"]

        # ================================================================
        # Initial threshold computation (lines 139-203)
        # ================================================================
        if reest_flag == 3:
            coh_thresh = np.float64(0.0)
            coh_thresh_coeffs = np.array([], dtype=np.float64)
        else:
            coh_thresh, coh_thresh_coeffs = self._compute_coh_threshold(
                coh_ps, D_A_all, D_A_bins, Nr_dist, coh_bins,
                low_coh_thresh, select_method, max_percent_rand,
            )

        # Ensure non-negative (line 205)
        if isinstance(coh_thresh, np.ndarray):
            coh_thresh[coh_thresh < 0] = 0.0
        else:
            coh_thresh = max(coh_thresh, 0.0)

        if isinstance(coh_thresh, np.ndarray):
            logger.info(
                "Initial gamma threshold: %.3f at D_A=%.2f to %.3f at D_A=%.2f",
                coh_thresh.min(), D_A_all.min(),
                coh_thresh.max(), D_A_all.max(),
            )
        else:
            logger.info("Initial gamma threshold: %.3f (uniform)", coh_thresh)

        # --- Initial PS selection (line 220) ---
        # MATLAB: ix = find(pm.coh_ps > coh_thresh)
        ix_mask = coh_ps > coh_thresh
        ix = np.where(ix_mask)[0]  # 0-based indices into original arrays
        n_selected = len(ix)
        logger.info("%d PS selected initially (from %d candidates)", n_selected, n_ps)

        # ================================================================
        # Reject part-time PS (lines 227-237)
        # ================================================================
        if gamma_stdev_reject > 0 and n_selected > 0:
            ix, n_selected = self._reject_parttime_ps(
                ix, ph_res_orig, ifg_index_0, gamma_stdev_reject,
            )

        # ================================================================
        # Re-estimation branch (lines 242-432)
        # ================================================================
        if reest_flag != 1:
            if reest_flag != 2:
                # Re-estimate coherence with PS removed from filtered patch
                logger.info("Re-estimating gamma for %d selected pixels...", n_selected)
                t_reest = time.time()

                # Log dropped ifgs
                for di in drop_ifg_index:
                    logger.info("ifg %d is dropped from noise re-estimation", int(di))

                # Subset data for selected pixels (ensure complex64: HDF5 may store as compound real/imag)
                ph_sel = _ensure_complex64(ph[ix, :])  # (n_selected, n_ifg)

                if isinstance(coh_thresh, np.ndarray) and len(coh_thresh) > 1:
                    coh_thresh = coh_thresh[ix]

                # Load bperp_mat for selected pixels
                bperp_mat_sel = self._load_bperp_mat(ix)

                n_i = int(grid_ij[:, 0].max())
                n_j = int(grid_ij[:, 1].max())

                # --- Per-pixel patch re-filtering (lines 270-317) ---
                n_win = int(parms["n_win"])
                slc_osf = int(parms["slc_osf"])
                clap_alpha = parms["clap_alpha"]
                clap_beta = parms["clap_beta"]

                selected_grid_ij = np.asarray(grid_ij[ix, :], dtype=np.int32)
                unique_grid_ij, inverse_grid = np.unique(
                    selected_grid_ij,
                    axis=0,
                    return_inverse=True,
                )
                n_unique_grid = len(unique_grid_ij)
                cache_hits = n_selected - n_unique_grid
                logger.info(
                    "Re-estimation uses %d unique grid cells (%.1f%% cache reuse)",
                    n_unique_grid,
                    cache_hits / max(n_selected, 1) * 100.0,
                )

                ph_patch_unique = np.zeros((n_unique_grid, n_ifg), dtype=np.complex64)

                # Pre-compute 7×7 Gaussian kernel once (used inside clap_filt_patch)
                gw7 = _gausswin(7).astype(np.float64)

                for i, (ps_i, ps_j) in enumerate(unique_grid_ij):
                    ps_i = int(ps_i)  # 1-based
                    ps_j = int(ps_j)

                    # Window bounds (1-based → convert to 0-based for slicing)
                    i_min_1 = max(ps_i - n_win // 2, 1)
                    i_max_1 = i_min_1 + n_win - 1
                    if i_max_1 > n_i:
                        i_min_1 = i_min_1 - i_max_1 + n_i
                        i_max_1 = n_i

                    j_min_1 = max(ps_j - n_win // 2, 1)
                    j_max_1 = j_min_1 + n_win - 1
                    if j_max_1 > n_j:
                        j_min_1 = j_min_1 - j_max_1 + n_j
                        j_max_1 = n_j

                    # Crude guard for small patches (MATLAB line 287)
                    if j_min_1 < 1 or i_min_1 < 1:
                        ph_patch_unique[i, :] = 0
                        continue

                    # 0-based slice bounds
                    i0 = i_min_1 - 1
                    i1 = i_max_1
                    j0 = j_min_1 - 1
                    j1 = j_max_1

                    # Position of PS pixel within window (0-based)
                    ps_bit_i = ps_i - i_min_1  # 0-based
                    ps_bit_j = ps_j - j_min_1

                    ph_bit = ph_grid[i0:i1, j0:j1, :].copy()
                    # Remove the pixel being tested (line 296)
                    ph_bit[ps_bit_i, ps_bit_j, :] = 0

                    # Oversampling removal (lines 301-305)
                    if slc_osf > 1:
                        ri = np.arange(
                            max(0, ps_bit_i - (slc_osf - 1)),
                            min(ph_bit.shape[0], ps_bit_i + slc_osf),
                        )
                        rj = np.arange(
                            max(0, ps_bit_j - (slc_osf - 1)),
                            min(ph_bit.shape[1], ps_bit_j + slc_osf),
                        )
                        ph_bit[np.ix_(ri, rj)] = 0

                    # Batch CLAP filter: process all IFGs at once using 3-D FFT.
                    # This replaces the inner for-loop over n_ifg.
                    # clap_filt_patch logic inlined for efficiency.
                    ph_bit[np.isnan(ph_bit)] = 0
                    ph_fft = np.fft.fft2(ph_bit, axes=(0, 1))  # (nw, nw, n_ifg)
                    H = np.abs(ph_fft)
                    # Smooth: MATLAB filter2(B, fftshift(H)) for all IFGs.
                    H = _smooth_spectral_magnitude(H, gw7)
                    med_H = np.median(H, axis=(0, 1), keepdims=True)  # (1,1,n_ifg)
                    med_H[med_H == 0] = 1.0
                    H = H / med_H
                    H = H ** clap_alpha
                    H = H - 1.0
                    H[H < 0] = 0.0
                    G = H * clap_beta + low_pass[:, :, np.newaxis]
                    ph_filtered = np.fft.ifft2(ph_fft * G, axes=(0, 1))
                    ph_patch_unique[i, :] = ph_filtered[ps_bit_i, ps_bit_j, :]

                    if (i + 1) % 2000 == 0 or (i + 1) == n_unique_grid:
                        elapsed_so_far = time.time() - t_reest
                        rate = (i + 1) / elapsed_so_far
                        eta = (n_unique_grid - i - 1) / rate if rate > 0 else 0
                        logger.info(
                            "%d / %d unique patches re-estimated (%.1f windows/s, ETA %.0f s)",
                            i + 1, n_unique_grid, rate, eta,
                        )

                ph_patch2 = ph_patch_unique[inverse_grid]

                logger.info(
                    "Patch re-estimation done: %.1f s (%d unique grid cells, %.1f%% cache hits)",
                    time.time() - t_reest,
                    n_unique_grid,
                    cache_hits / max(n_selected, 1) * 100.0,
                )

                # --- Batch topofit for all selected pixels (lines 324-340) ---
                logger.info("Re-estimating coherences via batch topofit...")
                t_topo = time.time()

                psdph = ph_sel * np.conj(ph_patch2)  # (n_selected, n_ifg)

                # Normalise (line 327): psdph = psdph ./ abs(psdph)
                abs_psdph = np.abs(psdph)
                abs_psdph[abs_psdph == 0] = 1.0
                psdph = psdph / abs_psdph

                # Run topofit only on ifg_index columns
                K_ps2, C_ps2, coh_ps2, ph_res2_ifg, _ = ps_topofit_batch(
                    psdph[:, ifg_index_0],
                    bperp_mat_sel[:, ifg_index_0],
                    n_trial_wraps,
                )

                # Place residual phase in full array
                ph_res2 = np.zeros((n_selected, n_ifg), dtype=np.float32)
                ph_res2[:, ifg_index_0] = ph_res2_ifg

                logger.info("Topofit re-estimation done: %.1f s", time.time() - t_topo)

            else:
                # reest_flag == 2: reuse previous select
                logger.info("Re-using previously calculated re-estimation (reest_flag=2)")
                (ix, K_ps2, C_ps2, coh_ps2,
                 ph_res2, ph_patch2) = self._load_prev_select()
                n_selected = len(ix)

            # --- Update coh_ps with re-estimated values (line 350) ---
            coh_ps_updated = coh_ps.copy()
            coh_ps_updated[ix] = coh_ps2

            # --- Recompute threshold with updated coherences (lines 353-412) ---
            coh_thresh, coh_thresh_coeffs = self._compute_coh_threshold(
                coh_ps_updated, D_A_all, D_A_bins, Nr_dist, coh_bins,
                low_coh_thresh, select_method, max_percent_rand,
            )

            if isinstance(coh_thresh, np.ndarray):
                coh_thresh[coh_thresh < 0] = 0.0
                # For re-est, threshold is only applied to selected pixels
                # MATLAB line 403: coh_thresh = polyval(…, D_A(ix))
                if len(coh_thresh) == len(D_A_all):
                    coh_thresh = coh_thresh[ix]
            else:
                coh_thresh = max(coh_thresh, 0.0)

            if isinstance(coh_thresh, np.ndarray):
                logger.info(
                    "Re-estimation gamma threshold: %.3f at D_A=%.2f to %.3f at D_A=%.2f",
                    coh_thresh.min(), D_A_all[ix].min() if len(ix) > 0 else 0,
                    coh_thresh.max(), D_A_all[ix].max() if len(ix) > 0 else 0,
                )
            else:
                logger.info("Re-estimation gamma threshold: %.3f (uniform)", coh_thresh)

            # --- Keep condition (line 419) ---
            # MATLAB: keep_ix = coh_ps2 > coh_thresh &
            #         abs(pm.K_ps(ix) - K_ps2) < 2*pi / bperp_range
            bperp_range = bperp_1d.max() - bperp_1d.min()
            K_diff = np.abs(K_ps_orig[ix] - K_ps2)
            keep_ix = (coh_ps2 > coh_thresh) & (K_diff < 2 * np.pi / bperp_range)

            n_kept = int(keep_ix.sum())
            logger.info("%d PS selected after re-estimation of coherence", n_kept)

        else:
            # reest_flag == 1: skip re-estimation (lines 424-432)
            logger.info("Skipping re-estimation (reest_flag=1)")
            ph_patch2 = ph_patch_orig[ix, :]
            ph_res2 = ph_res_orig[ix, :]
            K_ps2 = K_ps_orig[ix]
            C_ps2 = C_ps_orig[ix]
            coh_ps2 = coh_ps[ix]
            keep_ix = np.ones(n_selected, dtype=bool)
            n_kept = n_selected

        # ================================================================
        # Update no_ps_info (lines 436-448)
        # ================================================================
        no_ps_path = self.patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = self._load_no_ps_info(no_ps_path)
        stamps_step_no_ps[2:] = 0
        if n_kept == 0:
            logger.warning("*** No PS points left after Step 3 ***")
            stamps_step_no_ps[2] = 1
        self._save_no_ps_info(no_ps_path, stamps_step_no_ps)

        # ================================================================
        # Save results (line 463)
        # ================================================================
        select_path = self.patch_dir / f"select{self.psver}.h5"
        save_ph_patch2 = _env_flag("STAMPS_SAVE_SELECT_PH_PATCH2", False)
        save_ph_res2 = _env_flag("STAMPS_SAVE_SELECT_PH_RES2", False)

        # Convert ix to 1-based for MATLAB compatibility
        ix_1based = ix.astype(np.int32) + 1

        _save_select_h5(
            select_path,
            ix=ix_1based,
            keep_ix=keep_ix,
            ph_patch2=ph_patch2,
            ph_res2=ph_res2,
            K_ps2=K_ps2,
            C_ps2=C_ps2,
            coh_ps2=coh_ps2,
            coh_thresh=np.atleast_1d(coh_thresh),
            coh_thresh_coeffs=coh_thresh_coeffs,
            clap_alpha=parms["clap_alpha"],
            clap_beta=parms["clap_beta"],
            n_win=int(parms["n_win"]),
            max_percent_rand=max_percent_rand,
            gamma_stdev_reject=gamma_stdev_reject,
            small_baseline_flag=parms["small_baseline_flag"],
            ifg_index=ifg_index_save,
            save_ph_patch2=save_ph_patch2,
            save_ph_res2=save_ph_res2,
        )

        elapsed = time.time() - t_start
        logger.info(
            "Step 3 complete: %d → %d PS (%.1f s)",
            n_ps, n_kept, elapsed,
        )

    # ------------------------------------------------------------------
    # Internal: load parameters
    # ------------------------------------------------------------------
    def _load_parameters(self) -> dict:
        """Read all parameters needed by ps_select from config.

        MATLAB correspondence (ps_select.m lines 49-61)::

            slc_osf           = getparm('slc_osf', 1)
            clap_alpha        = getparm('clap_alpha', 1)
            clap_beta         = getparm('clap_beta', 1)
            n_win             = getparm('clap_win', 1)
            select_method     = getparm('select_method', 1)
            percent_rand      = getparm('percent_rand', 1)
            density_rand      = getparm('density_rand', 1)
            gamma_stdev_reject = getparm('gamma_stdev_reject', 1)
            small_baseline_flag = getparm('small_baseline_flag', 1)
            drop_ifg_index    = getparm('drop_ifg_index', 1)
        """
        cfg = self._cfg
        drop_raw = cfg.getparm("drop_ifg_index")
        drop_ifg_index = _parse_drop_ifg_index(drop_raw)

        return {
            "slc_osf": float(cfg.getparm("slc_osf")),
            "clap_alpha": float(cfg.getparm("clap_alpha")),
            "clap_beta": float(cfg.getparm("clap_beta")),
            "n_win": int(cfg.getparm("clap_win")),
            "select_method": str(cfg.getparm("select_method")).strip(),
            "percent_rand": float(cfg.getparm("percent_rand")),
            "density_rand": float(cfg.getparm("density_rand")),
            "gamma_stdev_reject": float(cfg.getparm("gamma_stdev_reject")),
            "small_baseline_flag": str(cfg.getparm("small_baseline_flag")).strip().lower(),
            "drop_ifg_index": drop_ifg_index,
        }

    # ------------------------------------------------------------------
    # Internal: load Step-1 data
    # ------------------------------------------------------------------
    def _load_ps_data(self, parms: dict):
        """Load Step-1 output (ps1.h5 or .mat).

        MATLAB correspondence (ps_select.m lines 79-104).
        """
        ps_h5 = self.patch_dir / f"ps{self.psver}.h5"
        if not ps_h5.is_file():
            raise FileNotFoundError(f"Step-1 output not found: {ps_h5}")

        logger.info("Loading Step-1 data from %s", ps_h5)

        with h5py.File(str(ps_h5), "r") as hf:
            n_ps = int(hf["n_ps"][()])
            n_ifg = int(hf["n_ifg"][()])
            master_ix = int(hf["master_ix"][()])  # 1-based
            bperp_1d = np.asarray(hf["bperp"][:], dtype=np.float64)
            xy = np.asarray(hf["xy"][:], dtype=np.float32)

            if "ifgday_ix" in hf:
                ifgday_ix = np.asarray(hf["ifgday_ix"][:], dtype=np.int32)
            else:
                ifgday_ix = None

            if "D_A" in hf:
                D_A = np.asarray(hf["D_A"][:], dtype=np.float64).ravel()
            else:
                D_A = np.array([], dtype=np.float64)

        # Load ph (prefer separate ph file)
        ph = self._load_ph(n_ps, n_ifg)

        return (ph, bperp_1d, n_ifg, n_ps, xy, master_ix, ifgday_ix, D_A)

    def _load_ph(self, n_ps: int, n_ifg: int) -> np.ndarray:
        """Load complex phase (ph1.h5 > ph1.mat > ps1.h5:/ph)."""
        ph_h5 = self.patch_dir / f"ph{self.psver}.h5"
        if ph_h5.is_file():
            with h5py.File(str(ph_h5), "r") as hf:
                return np.asarray(hf["ph"][:])

        ph_mat = self.patch_dir / f"ph{self.psver}.mat"
        if ph_mat.is_file():
            return np.asarray(load_mat(ph_mat)["ph"])

        ps_h5 = self.patch_dir / f"ps{self.psver}.h5"
        with h5py.File(str(ps_h5), "r") as hf:
            if "ph" in hf:
                return np.asarray(hf["ph"][:])

        raise FileNotFoundError(f"Complex phase not found in {self.patch_dir}")

    # ------------------------------------------------------------------
    # Internal: load Step-2 data
    # ------------------------------------------------------------------
    def _load_pm_data(self):
        """Load Step-2 output (pm1.h5 or pm1.mat).

        MATLAB correspondence (ps_select.m line 106).
        """
        # Try h5 first, then mat
        pm_h5 = self.patch_dir / f"pm{self.psver}.h5"
        pm_mat = self.patch_dir / f"pm{self.psver}.mat"

        if pm_h5.is_file():
            logger.info("Loading Step-2 data from %s", pm_h5.name)
            with h5py.File(str(pm_h5), "r") as hf:
                coh_ps = np.asarray(hf["coh_ps"][:], dtype=np.float64).ravel()
                coh_bins = np.asarray(hf["coh_bins"][:], dtype=np.float64).ravel()
                Nr_dist = np.asarray(hf["Nr"][:], dtype=np.float64).ravel()
                K_ps = np.asarray(hf["K_ps"][:], dtype=np.float64).ravel()
                C_ps = np.asarray(hf["C_ps"][:], dtype=np.float64).ravel()
                ph_res = np.asarray(hf["ph_res"][:], dtype=np.float32)
                ph_patch = np.asarray(hf["ph_patch"][:])
                ph_grid = np.asarray(hf["ph_grid"][:])
                ph_weight = np.asarray(hf["ph_weight"][:]) if "ph_weight" in hf else None
                n_trial_wraps = float(hf["n_trial_wraps"][()])
                grid_ij = np.asarray(hf["grid_ij"][:], dtype=np.float32)
                low_pass = np.asarray(hf["low_pass"][:], dtype=np.float64)
        elif pm_mat.is_file():
            logger.info("Loading Step-2 data from %s", pm_mat.name)
            d = load_mat(pm_mat)
            coh_ps = np.asarray(d["coh_ps"]).ravel().astype(np.float64)
            coh_bins = np.asarray(d["coh_bins"]).ravel().astype(np.float64)
            Nr_dist = np.asarray(d["Nr"]).ravel().astype(np.float64)
            K_ps = np.asarray(d["K_ps"]).ravel().astype(np.float64)
            C_ps = np.asarray(d["C_ps"]).ravel().astype(np.float64)
            ph_res = np.asarray(d["ph_res"]).astype(np.float32)
            ph_patch = np.asarray(d["ph_patch"])
            ph_grid = np.asarray(d["ph_grid"])
            ph_weight = np.asarray(d["ph_weight"])
            n_trial_wraps = float(np.asarray(d["n_trial_wraps"]).ravel()[0])
            grid_ij = np.asarray(d["grid_ij"]).astype(np.float32)
            low_pass = np.asarray(d["low_pass"]).astype(np.float64)
        else:
            raise FileNotFoundError(
                f"Step-2 output not found in {self.patch_dir} "
                f"(checked pm{self.psver}.h5 and pm{self.psver}.mat)"
            )

        return (coh_ps, coh_bins, Nr_dist, K_ps, C_ps,
                ph_res, ph_patch, ph_grid, ph_weight,
                n_trial_wraps, grid_ij, low_pass)

    # ------------------------------------------------------------------
    # Internal: load bperp_mat for selected pixels
    # ------------------------------------------------------------------
    def _load_bperp_mat(self, ix: np.ndarray) -> np.ndarray:
        """Load bperp_mat for selected pixel indices.

        MATLAB correspondence (ps_select.m lines 320-322)::

            bp = load(bpname);
            bperp_mat = bp.bperp_mat(ix, :);
        """
        # Try bp1.h5
        bp_h5 = self.patch_dir / f"bp{self.psver}.h5"
        if bp_h5.is_file():
            with h5py.File(str(bp_h5), "r") as hf:
                ds = hf["bperp_mat"]
                if len(ix) > max(10000, int(ds.shape[0] * 0.05)):
                    return np.asarray(ds[:, :], dtype=np.float32)[ix, :]
                # h5py fancy indexing is efficient only for small selections.
                sorted_ix = np.sort(ix)
                bperp_full = np.asarray(ds[sorted_ix, :], dtype=np.float32)
                inv_order = np.argsort(np.argsort(ix))
                return bperp_full[inv_order]

        # Try bp1.mat
        bp_mat = self.patch_dir / f"bp{self.psver}.mat"
        if bp_mat.is_file():
            d = load_mat(bp_mat)
            return np.asarray(d["bperp_mat"])[ix, :].astype(np.float32)

        # Fall back to ps1.h5
        ps_h5 = self.patch_dir / f"ps{self.psver}.h5"
        with h5py.File(str(ps_h5), "r") as hf:
            if "bperp_mat" in hf:
                ds = hf["bperp_mat"]
                if len(ix) > max(10000, int(ds.shape[0] * 0.05)):
                    return np.asarray(ds[:, :], dtype=np.float32)[ix, :]
                sorted_ix = np.sort(ix)
                bperp_full = np.asarray(ds[sorted_ix, :], dtype=np.float32)
                inv_order = np.argsort(np.argsort(ix))
                return bperp_full[inv_order]

        raise FileNotFoundError(
            f"bperp_mat not found in {self.patch_dir}"
        )

    # ------------------------------------------------------------------
    # Internal: D_A binning
    # ------------------------------------------------------------------
    def _compute_da_bins(
        self, D_A: np.ndarray, coh_ps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute D_A bin edges.

        MATLAB correspondence (ps_select.m lines 115-128)::

            if ~isempty(D_A) & size(D_A,1) >= 10000
                D_A_sort = sort(D_A);
                if size(D_A,1) >= 50000
                    bin_size = 10000;
                else
                    bin_size = 2000;
                end
                D_A_max = [0; D_A_sort(bin_size:bin_size:end-bin_size); D_A_sort(end)];
            else
                D_A_max = [0; 1];
                D_A = ones(size(pm.coh_ps));
            end

        Returns
        -------
        D_A_bins : (n_bins+1,) — bin edges [0, …, D_A_max]
        D_A_all : (n_ps,) — D_A array (may be replaced with ones)
        """
        if len(D_A) >= 10000:
            D_A_sort = np.sort(D_A)
            bin_size = 10000 if len(D_A) >= 50000 else 2000
            # MATLAB: D_A_sort(bin_size:bin_size:end-bin_size)
            # 1-based indices are bin_size, 2*bin_size, ..., <= n-bin_size.
            # Convert to 0-based by subtracting 1.
            mid_ix = np.arange(bin_size - 1, len(D_A_sort) - bin_size, bin_size)
            mid_edges = D_A_sort[mid_ix]
            D_A_bins = np.concatenate([[0.0], mid_edges, [D_A_sort[-1]]])
            return D_A_bins, D_A
        else:
            D_A_bins = np.array([0.0, 1.0])
            D_A_ones = np.ones(len(coh_ps), dtype=np.float64)
            return D_A_bins, D_A_ones

    # ------------------------------------------------------------------
    # Internal: compute coherence threshold from histogram analysis
    # ------------------------------------------------------------------
    def _compute_coh_threshold(
        self,
        coh_ps: np.ndarray,
        D_A: np.ndarray,
        D_A_bins: np.ndarray,
        Nr_dist: np.ndarray,
        coh_bins: np.ndarray,
        low_coh_thresh: int,
        select_method: str,
        max_percent_rand: float,
    ) -> Tuple[Union[float, np.ndarray], Optional[np.ndarray]]:
        """Compute per-pixel or uniform coherence threshold.

        MATLAB correspondence (ps_select.m lines 143-203).

        Returns
        -------
        coh_thresh : float or ndarray(n_ps,) — threshold per pixel
        coh_thresh_coeffs : ndarray(2,) or None — linear fit coefficients
        """
        n_bins = len(D_A_bins) - 1
        min_coh = np.zeros(n_bins, dtype=np.float64)
        D_A_mean = np.zeros(n_bins, dtype=np.float64)

        for i in range(n_bins):
            mask = (D_A > D_A_bins[i]) & (D_A <= D_A_bins[i + 1])
            coh_chunk = coh_ps[mask]
            D_A_mean[i] = float(np.mean(D_A[mask])) if mask.sum() > 0 else 0.0

            min_coh[i] = _compute_coh_threshold_single_bin(
                coh_chunk, coh_bins, Nr_dist, low_coh_thresh,
                select_method, max_percent_rand,
            )

        # Remove NaN bins (lines 182-189)
        valid = ~np.isnan(min_coh)
        if valid.sum() < 1:
            logger.warning(
                "Not enough random phase pixels to set gamma threshold — "
                "using default threshold of 0.3"
            )
            return 0.3, np.array([], dtype=np.float64)

        min_coh_valid = min_coh[valid]
        D_A_mean_valid = D_A_mean[valid]

        if len(min_coh_valid) > 1:
            # Linear fit: coh_thresh_coeffs = polyfit(D_A_mean, min_coh, 1)
            coh_thresh_coeffs = np.polyfit(D_A_mean_valid, min_coh_valid, 1)

            if coh_thresh_coeffs[0] > 0:
                # Positive slope (expected)
                coh_thresh = np.polyval(coh_thresh_coeffs, D_A)
            else:
                # Negative slope → use average threshold
                coh_thresh = float(np.polyval(coh_thresh_coeffs, 0.35))
                coh_thresh_coeffs = np.array([], dtype=np.float64)
        else:
            coh_thresh = float(min_coh_valid[0])
            coh_thresh_coeffs = np.array([], dtype=np.float64)

        return coh_thresh, coh_thresh_coeffs

    # ------------------------------------------------------------------
    # Internal: reject part-time PS
    # ------------------------------------------------------------------
    def _reject_parttime_ps(
        self,
        ix: np.ndarray,
        ph_res: np.ndarray,
        ifg_index_0: np.ndarray,
        gamma_stdev_reject: float,
    ) -> Tuple[np.ndarray, int]:
        """Reject PS with high coherence standard deviation (bootstrap).

        MATLAB correspondence (ps_select.m lines 227-237)::

            ph_res_cpx = exp(j * pm.ph_res(:, ifg_index));
            for i = 1:length(ix)
                coh_std(i) = std(bootstrp(100, @(ph) abs(sum(ph))/length(ph),
                                          ph_res_cpx(ix(i), ifg_index)));
            end
            ix = ix(coh_std < gamma_stdev_reject);

        NOTE: This is vectorised using numpy bootstrap.
        """
        logger.info("Rejecting part-time PS (gamma_stdev_reject=%.3f)...", gamma_stdev_reject)
        n_boot = 100
        ph_res_cpx = np.exp(1j * ph_res[ix][:, ifg_index_0])  # (n_sel, n_ifg_used)
        n_sel, n_ifg_used = ph_res_cpx.shape

        rng = np.random.RandomState(42)
        # Bootstrap: draw n_boot samples of size n_ifg_used with replacement
        boot_idx = rng.randint(0, n_ifg_used, size=(n_boot, n_ifg_used))

        # For each pixel: compute 100 coherence estimates, take std
        coh_std = np.zeros(n_sel, dtype=np.float64)
        for i in range(n_sel):
            pixel_cpx = ph_res_cpx[i, :]  # (n_ifg_used,)
            boot_samples = pixel_cpx[boot_idx]  # (n_boot, n_ifg_used)
            boot_coh = np.abs(np.sum(boot_samples, axis=1)) / n_ifg_used
            coh_std[i] = np.std(boot_coh, ddof=1)

        keep = coh_std < gamma_stdev_reject
        ix = ix[keep]
        logger.info("%d PS left after part-time PS rejection", len(ix))
        return ix, len(ix)

    # ------------------------------------------------------------------
    # Internal: load previous select for reest_flag==2
    # ------------------------------------------------------------------
    def _load_prev_select(self):
        """Load previous select1 output for reuse."""
        sel_h5 = self.patch_dir / f"select{self.psver}.h5"
        sel_mat = self.patch_dir / f"select{self.psver}.mat"

        if sel_h5.is_file():
            with h5py.File(str(sel_h5), "r") as hf:
                if "ph_patch2" not in hf:
                    raise FileNotFoundError(
                        f"{sel_h5.name} does not contain ph_patch2. "
                        "Re-run Step 3 with reest_flag=0, or provide a MATLAB select*.mat "
                        "file for reest_flag=2 reuse."
                    )
                ix = np.asarray(hf["ix"][:], dtype=np.int64).ravel() - 1  # to 0-based
                K_ps2 = np.asarray(hf["K_ps2"][:], dtype=np.float64).ravel()
                C_ps2 = np.asarray(hf["C_ps2"][:], dtype=np.float64).ravel()
                coh_ps2 = np.asarray(hf["coh_ps2"][:], dtype=np.float64).ravel()
                if "ph_res2" not in hf:
                    raise FileNotFoundError(
                        f"{sel_h5.name} does not contain ph_res2. "
                        "Re-run Step 3 with STAMPS_SAVE_SELECT_PH_RES2=1, or provide "
                        "a MATLAB select*.mat file for reest_flag=2 reuse."
                    )
                ph_res2 = np.asarray(hf["ph_res2"][:], dtype=np.float32)
                ph_patch2 = np.asarray(hf["ph_patch2"][:])
            return ix, K_ps2, C_ps2, coh_ps2, ph_res2, ph_patch2

        if sel_mat.is_file():
            d = load_mat(sel_mat)
            ix = np.asarray(d["ix"]).ravel().astype(np.int64) - 1  # to 0-based
            ph_res2 = np.asarray(d["ph_res2"], dtype=np.float32)
            ph_patch2 = _ensure_complex64(np.asarray(d["ph_patch2"]))
            if ph_res2.ndim == 2 and ph_res2.shape[0] != ix.size and ph_res2.shape[1] == ix.size:
                ph_res2 = np.ascontiguousarray(ph_res2.T)
            if ph_patch2.ndim == 2 and ph_patch2.shape[0] != ix.size and ph_patch2.shape[1] == ix.size:
                ph_patch2 = np.ascontiguousarray(ph_patch2.T)
            return (ix, np.asarray(d["K_ps2"]).ravel(), np.asarray(d["C_ps2"]).ravel(),
                    np.asarray(d["coh_ps2"]).ravel(), ph_res2, ph_patch2)

        raise FileNotFoundError(f"Previous select output not found in {self.patch_dir}")

    # ------------------------------------------------------------------
    # Internal: no_ps_info helpers
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

def ps_select(
    patch_dir: Union[str, Path],
    reest_flag: int = 0,
    psver: int = 1,
) -> None:
    """Module-level entry point (mirrors MATLAB ``ps_select``).

    Parameters
    ----------
    patch_dir : str or Path
        Patch directory containing ``ps1.h5``, ``pm1.h5``, etc.
    reest_flag : int
        0 = re-estimate (default), 1 = skip, 2 = reuse, 3 = all.
    psver : int
        PS version number.
    """
    selector = PSSelector(patch_dir=patch_dir, psver=psver)
    selector.run(reest_flag=reest_flag)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 3: Select PS pixels (ps_select)"
    )
    parser.add_argument(
        "patch_dir", type=str,
        help="Path to patch directory containing ps1.h5 and pm1.h5",
    )
    parser.add_argument(
        "--reest-flag", type=int, default=0,
        help="Re-estimation flag (0=reest, 1=skip, 2=reuse, 3=all)",
    )
    parser.add_argument(
        "--psver", type=int, default=1,
        help="PS version number (default 1)",
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

    ps_select(args.patch_dir, reest_flag=args.reest_flag, psver=args.psver)
