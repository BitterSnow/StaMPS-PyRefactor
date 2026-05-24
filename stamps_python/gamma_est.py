#!/usr/bin/env python3
"""
gamma_est.py — Python port of ps_est_gamma_quick.m  (StaMPS Step 2)
====================================================================

Estimate coherence (gamma) for PS candidate pixels.

MATLAB → Python mapping
-----------------------
+-----------------------------------+-------------------------------------------------+
| MATLAB concept                    | Python mapping                                  |
+===================================+=================================================+
| ``ps_est_gamma_quick(restart)``   | ``GammaEstimator(patch_dir).run(restart)``      |
+-----------------------------------+-------------------------------------------------+
| ``ps_topofit(cpx, bperp, …)``    | ``ps_topofit_batch`` (vectorised, no pixel loop)|
+-----------------------------------+-------------------------------------------------+
| ``clap_filt(ph, α, β, …)``       | ``clap_filt`` (pure-Python port)                |
+-----------------------------------+-------------------------------------------------+
| ``ph_grid`` 3-D accumulation loop| ``np.add.at`` on flattened grid index           |
+-----------------------------------+-------------------------------------------------+
| ``ph_patch = ph_filt(ij…)``      | Advanced NumPy fancy indexing                   |
+-----------------------------------+-------------------------------------------------+
| ``rand('state', 2005)``          | ``np.random.RandomState(2005)``                 |
+-----------------------------------+-------------------------------------------------+
| ``stamps_save('pm1', …)``        | ``_save_pm_h5(…, pm1.h5)``                     |
+-----------------------------------+-------------------------------------------------+
| ``setappdata / getappdata``       | Data persisted in HDF5                          |
+-----------------------------------+-------------------------------------------------+

Numeric type conventions
------------------------
* Phase arrays (``ph``, ``ph_patch``, ``ph_weight``) are ``complex64`` to
  match MATLAB ``single`` and to halve memory.
* ``K_ps``, ``C_ps``, ``coh_ps`` are ``float64`` (MATLAB ``double``).
* ``np.std(…, ddof=1)`` matches MATLAB ``std()`` default.
* Per-pixel baselines ``bperp_mat`` are ``float32`` (same as MATLAB).

Performance notes
-----------------
* **No explicit pixel loops.**  Grid accumulation uses ``np.add.at``;
  grid extraction uses fancy indexing; ``ps_topofit_batch`` processes all
  pixels (or a chunk) in one call with vectorised NumPy operations.
* Large matrices (``ph``, ``bperp_mat``) are read from HDF5 with slicing
  support so chunk-wise processing can be added for memory-constrained
  systems.
* The CLAP filter loops over *spatial windows* (typically < 100 iterations
  for a small grid), not over pixels.

Refactored from: ps_est_gamma_quick.m (Andy Hooper, June 2006)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import gaussian as _scipy_gaussian

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig, load_mat  # noqa: E402

logger = logging.getLogger("stamps")
_FAST_COMPRESSION = {}


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ============================================================================
# Small helper utilities
# ============================================================================

def _gausswin(n: int, alpha: float = 2.5) -> np.ndarray:
    """Replicate MATLAB ``gausswin(N, alpha)``.

    MATLAB: ``w(k) = exp(-0.5 * (alpha * 2*(k - (N-1)/2) / (N-1))^2)``
    which is equivalent to ``scipy.signal.windows.gaussian(N, std=(N-1)/(2*alpha))``.
    """
    std = (n - 1) / (2 * alpha)
    return _scipy_gaussian(n, std=std)


def _butterworth_lowpass(n_win: int, grid_size: float,
                         low_pass_wavelength: float) -> np.ndarray:
    """Build the 2-D Butterworth low-pass filter used by CLAP.

    MATLAB correspondence (ps_est_gamma_quick.m lines 54-58)::

        freq0 = 1 / low_pass_wavelength;
        freq_i = -(n_win)/grid_size/n_win/2 : 1/grid_size/n_win
                 : (n_win-2)/grid_size/n_win/2;
        butter_i = 1 ./ (1 + (freq_i / freq0).^(2*5));
        low_pass = butter_i' * butter_i;
        low_pass = fftshift(low_pass);

    Returns
    -------
    low_pass : ndarray, shape (n_win, n_win)
        fftshift-ed 2-D Butterworth filter.
    """
    freq0 = 1.0 / low_pass_wavelength
    # MATLAB: start =  -n_win / (grid_size * n_win * 2)
    #         stop  = (n_win - 2) / (grid_size * n_win * 2)
    #         step  = 1 / (grid_size * n_win)
    start = -n_win / (grid_size * n_win * 2)
    step = 1.0 / (grid_size * n_win)
    stop = (n_win - 2) / (grid_size * n_win * 2) + step * 0.5  # inclusive
    freq_i = np.arange(start, stop, step)
    # Ensure exactly n_win elements (numerical precision guard)
    freq_i = freq_i[:n_win]

    butter_i = 1.0 / (1.0 + (freq_i / freq0) ** (2 * 5))
    low_pass = np.outer(butter_i, butter_i)
    low_pass = np.fft.fftshift(low_pass)
    return low_pass


# ============================================================================
# CLAP filter  —  port of clap_filt.m
# ============================================================================

def clap_filt(ph: np.ndarray,
              alpha: float = 1.0,
              beta: float = 0.3,
              n_win: int = 24,
              n_pad: int = 8,
              low_pass: Optional[np.ndarray] = None) -> np.ndarray:
    """Combined Low-pass Adaptive Phase (CLAP) filtering.

    Port of ``clap_filt.m`` (Andy Hooper, June 2006).

    The function applies overlapping-window adaptive spectral filtering
    to a 2-D complex phase grid.  For each window:

    1. Compute the 2-D FFT of the zero-padded window tile.
    2. Estimate the spectral magnitude, smooth it with a 7×7 Gaussian.
    3. Raise to power *alpha*, subtract median → adaptive weight H.
    4. Combine: ``G = H * beta + low_pass``.
    5. Apply G in the frequency domain and inverse-FFT.
    6. Overlap-add with a triangular (tent) weighting function.

    Parameters match the MATLAB call in ps_est_gamma_quick.m line 225::

        ph_filt(:,:,i) = clap_filt(ph_grid(:,:,i), clap_alpha, clap_beta,
                                    n_win*0.75, n_win*0.25, low_pass);

    Parameters
    ----------
    ph : ndarray, shape (n_i, n_j), complex
        2-D phase grid for one interferogram.
    alpha, beta : float
        CLAP parameters (``clap_alpha``, ``clap_beta``).
    n_win : int
        Window size (after 0.75 scaling of ``clap_win``).
    n_pad : int
        Zero-padding size (after 0.25 scaling of ``clap_win``).
    low_pass : ndarray or None
        Pre-computed Butterworth low-pass (fftshift-ed), shape ``(n_win+n_pad, n_win+n_pad)``.

    Returns
    -------
    ph_out : ndarray, same shape as *ph*, complex
    """
    n_win = int(n_win)
    n_pad = int(n_pad)
    n_win_ex = n_win + n_pad

    if low_pass is None:
        low_pass = np.zeros((n_win_ex, n_win_ex), dtype=np.float64)

    n_i, n_j = ph.shape
    ph_out = np.zeros_like(ph)

    n_inc = n_win // 4
    n_win_i = int(np.ceil(n_i / n_inc)) - 3
    n_win_j = int(np.ceil(n_j / n_inc)) - 3

    # Triangular (tent) weighting — MATLAB lines 38-43
    x = np.arange(n_win // 2)
    X, Y = np.meshgrid(x, x)
    XY = X + Y
    wind_func = np.block([[XY, np.fliplr(XY)],
                          [np.flipud(XY), np.flipud(np.fliplr(XY))]])
    wind_func = wind_func.astype(np.float64) + 1e-6

    # Replace NaN with 0 — MATLAB line 46
    ph = np.copy(ph)
    ph[np.isnan(ph)] = 0

    # 7×7 Gaussian smoothing kernel — MATLAB: B = gausswin(7)*gausswin(7)'
    gw7 = _gausswin(7)
    B = np.outer(gw7, gw7)

    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=ph.dtype)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack([np.zeros((i_shift, n_win)), wf[:n_win - i_shift, :]])

        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.hstack([np.zeros((n_win, j_shift)), wf2[:, :n_win - j_shift]])

            ph_bit[:, :] = 0
            ph_bit[:n_win, :n_win] = ph[i1:i2, j1:j2]

            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)

            # Smooth spectral magnitude — MATLAB line 74:
            #   H = ifftshift(filter2(B, fftshift(H)))
            H_shifted = np.fft.fftshift(H)
            # filter2(B, X) in MATLAB ≡ convolve2d(X, B, 'same')
            # Using fftconvolve for speed
            H_smooth = fftconvolve(H_shifted, B, mode="same")
            H = np.fft.ifftshift(H_smooth)

            # Normalise by median — MATLAB lines 75-78
            median_H = np.median(H)
            if median_H != 0:
                H = H / median_H

            H = H ** alpha
            H = H - 1.0
            H[H < 0] = 0.0

            G = H * beta + low_pass
            ph_filt = np.fft.ifft2(ph_fft * G)
            ph_filt = ph_filt[:n_win, :n_win] * wf2
            ph_out[i1:i2, j1:j2] += ph_filt

    return ph_out


# ============================================================================
# Vectorised ps_topofit  —  port of ps_topofit.m
# ============================================================================

def ps_topofit_batch(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    asym: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch-vectorised topographic-error fitting for *all* pixels at once.

    This replaces the per-pixel ``for i=1:n_ps`` loop in
    ``ps_est_gamma_quick.m`` lines 244-260, calling ``ps_topofit.m`` inside.

    **No explicit pixel loop is used.**  Instead, the grid search and linear
    refinement are expressed as broadcasted NumPy array operations.

    Parameters
    ----------
    cpxphase : ndarray, shape (N, n_ifg), complex
        Differential phase (``ph * conj(ph_patch)``).  Rows with *any* zero
        entry are treated as invalid and receive ``K=nan, coh=0``.
    bperp : ndarray, shape (N, n_ifg), float
        Per-pixel perpendicular baselines (``bperp_mat``).
    n_trial_wraps : float
        Number of expected trial wraps (``bperp_range * max_K / 2π``).
    asym : float
        Asymmetry parameter (default 0 → symmetric search).

    Returns
    -------
    K_ps : (N,) float64 — topographic phase rate
    C_ps : (N,) float64 — constant phase offset
    coh_ps : (N,) float64 — coherence
    ph_res : (N, n_ifg) float32 — residual phase angle
    N_opt : (N,) int32 — always 1 for valid pixels (single peak selected)

    MATLAB correspondence
    ---------------------
    ``ps_topofit.m`` variables are annotated inline below.
    """
    N, n_ifg = cpxphase.shape

    # Identify valid pixels: all interferograms non-zero
    # MATLAB (line 246): if sum(psdph==0)==0
    valid = np.all(cpxphase != 0, axis=1)  # (N,)

    # Pre-allocate outputs
    K_ps = np.full(N, np.nan, dtype=np.float64)
    C_ps = np.zeros(N, dtype=np.float64)
    coh_ps = np.zeros(N, dtype=np.float64)
    ph_res = np.zeros((N, n_ifg), dtype=np.float32)
    N_opt = np.zeros(N, dtype=np.int32)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return K_ps, C_ps, coh_ps, ph_res, N_opt

    # Extract valid subset
    cpx_v = cpxphase[valid]        # (n_valid, n_ifg)
    bp_v = bperp[valid]            # (n_valid, n_ifg)

    # --- Per-pixel bperp_range  (ps_topofit.m line 27) ---
    bp_max = bp_v.max(axis=1)     # (n_valid,)
    bp_min = bp_v.min(axis=1)
    bp_range = bp_max - bp_min    # (n_valid,)
    bp_range[bp_range == 0] = 1.0  # guard against zero range

    # --- Trial multipliers  (ps_topofit.m line 31) ---
    half_range = int(np.ceil(8 * n_trial_wraps))
    trial_mult = np.arange(-half_range, half_range + 1) + asym * 8 * n_trial_wraps
    trial_mult = trial_mult.astype(np.float64)
    n_trials = len(trial_mult)

    # --- Normalised baseline phase  (ps_topofit.m line 33) ---
    #   trial_phase = bperp / bperp_range * pi/4   →  (n_valid, n_ifg)
    norm_bp = bp_v / bp_range[:, None] * (np.pi / 4.0)

    # --- Grid search: loop over (small number of) trial multipliers ---
    # We accumulate phaser_sum for each trial without building a full 3-D array
    # MATLAB: phaser_sum = sum(exp(-j*trial_phase*trial_mult) .* cpxphase)
    abs_cpx_sum = np.sum(np.abs(cpx_v), axis=1)  # (n_valid,)
    abs_cpx_sum[abs_cpx_sum == 0] = 1.0           # guard

    best_coh = np.full(n_valid, -1.0, dtype=np.float64)
    best_idx = np.zeros(n_valid, dtype=np.int64)
    best_C = np.zeros(n_valid, dtype=np.float64)

    for t_idx in range(n_trials):
        t = trial_mult[t_idx]
        # trial_phase_mat column = exp(-j * norm_bp * t)   →  (n_valid, n_ifg)
        trial_ph = np.exp(-1j * norm_bp * t)
        phaser_sum = np.sum(trial_ph * cpx_v, axis=1)  # (n_valid,)
        coh_t = np.abs(phaser_sum) / abs_cpx_sum
        improved = coh_t > best_coh
        best_coh[improved] = coh_t[improved]
        best_idx[improved] = t_idx
        best_C[improved] = np.angle(phaser_sum[improved])

    # --- Best K from grid search  (ps_topofit.m line 43) ---
    # K0 = pi/4 / bperp_range * trial_mult(best_idx)
    K0 = (np.pi / 4.0) / bp_range * trial_mult[best_idx]

    # ===== Linear refinement  (ps_topofit.m lines 49-58) =====
    # resphase = cpxphase .* exp(-j * K0 * bperp)
    resphase = cpx_v * np.exp(-1j * K0[:, None] * bp_v)
    offset_phase = np.sum(resphase, axis=1)  # (n_valid,)
    # resphase = angle(resphase * conj(offset_phase))
    resphase_ang = np.angle(resphase * np.conj(offset_phase[:, None]))

    # Weighted least-squares:  mopt = (w.*bperp) \ (w.*resphase_ang)
    #   w = abs(cpxphase),  but phase is already unit-normalised → w ≈ 1
    w = np.abs(cpx_v)  # (n_valid, n_ifg)
    wb = w * bp_v
    wr = w * resphase_ang
    denom = np.sum(wb * wb, axis=1)
    denom[denom == 0] = 1.0
    mopt = np.sum(wb * wr, axis=1) / denom
    K0 = K0 + mopt

    # Final residual
    phase_residual = cpx_v * np.exp(-1j * K0[:, None] * bp_v)
    mean_pr = np.sum(phase_residual, axis=1)
    C0 = np.angle(mean_pr)
    coh0 = np.abs(mean_pr) / np.sum(np.abs(phase_residual), axis=1)

    # Write back to full arrays
    K_ps[valid] = K0
    C_ps[valid] = C0
    coh_ps[valid] = coh0
    ph_res[valid] = np.angle(phase_residual).astype(np.float32)
    N_opt[valid] = 1

    return K_ps, C_ps, coh_ps, ph_res, N_opt


def _ps_topofit_rand_batch(
    rand_cpx: np.ndarray,
    bperp_1d: np.ndarray,
    n_trial_wraps: float,
) -> np.ndarray:
    """Vectorised ps_topofit for the random-phase simulation.

    All random samples share the *same* bperp vector, which makes the grid
    search a simple matrix multiply ``rand_cpx @ trial_phase_mat``.

    Parameters
    ----------
    rand_cpx : (n_rand, n_ifg) complex64
    bperp_1d : (n_ifg,) float64
    n_trial_wraps : float

    Returns
    -------
    coh_rand : (n_rand,) float64  — random coherence for each sample
    """
    n_rand, n_ifg = rand_cpx.shape

    bp_range = bperp_1d.max() - bperp_1d.min()
    if bp_range == 0:
        return np.zeros(n_rand, dtype=np.float64)

    # Trial multipliers  (ps_topofit.m line 31)
    half_range = int(np.ceil(8 * n_trial_wraps))
    trial_mult = np.arange(-half_range, half_range + 1, dtype=np.float64)

    # trial_phase = bperp / bperp_range * pi/4   →  (n_ifg,)
    norm_bp = bperp_1d / bp_range * (np.pi / 4.0)

    # trial_phase_mat = exp(-j * norm_bp[:, None] * trial_mult[None, :])
    #   shape (n_ifg, n_trials)
    trial_phase_mat = np.exp(-1j * np.outer(norm_bp, trial_mult))

    # phaser_sum = rand_cpx @ trial_phase_mat   →  (n_rand, n_trials)
    phaser_sum = rand_cpx @ trial_phase_mat

    # abs sum per sample  (all |cpx| = 1 since rand_cpx = exp(j*angle))
    abs_cpx_sum = np.sum(np.abs(rand_cpx), axis=1)  # (n_rand,)
    abs_cpx_sum[abs_cpx_sum == 0] = 1.0

    coh_trial = np.abs(phaser_sum) / abs_cpx_sum[:, None]  # (n_rand, n_trials)

    # Best trial per sample
    best_idx = np.argmax(coh_trial, axis=1)
    K0 = (np.pi / 4.0) / bp_range * trial_mult[best_idx]

    # --- Linear refinement (same algebra as ps_topofit_batch) ---
    resphase = rand_cpx * np.exp(-1j * K0[:, None] * bperp_1d[None, :])
    offset_phase = np.sum(resphase, axis=1)
    resphase_ang = np.angle(resphase * np.conj(offset_phase[:, None]))

    w = np.abs(rand_cpx)
    wb = w * bperp_1d[None, :]
    wr = w * resphase_ang
    denom = np.sum(wb * wb, axis=1)
    denom[denom == 0] = 1.0
    mopt = np.sum(wb * wr, axis=1) / denom
    K0 = K0 + mopt

    phase_residual = rand_cpx * np.exp(-1j * K0[:, None] * bperp_1d[None, :])
    mean_pr = np.sum(phase_residual, axis=1)
    coh0 = np.abs(mean_pr) / np.sum(np.abs(phase_residual), axis=1)

    return coh0


# ============================================================================
# Utility: read incidence angle
# ============================================================================

def _load_inc_mean(patch_dir: Path) -> float:
    """Load mean incidence angle from inc1 or la1 files.

    MATLAB correspondence (ps_est_gamma_quick.m lines 118-132)::

        if exist(incname, 'file')
            inc = load(incname);
            inc_mean = mean(inc.inc(inc.inc~=0));
        elseif exist(laname, 'file')
            la = load(laname);
            inc_mean = mean(la.la) + 0.052;   % +3 deg
        else
            inc_mean = 21 * pi / 180;
        end
    """
    # Try inc1.h5
    inc_h5 = patch_dir / "inc1.h5"
    if inc_h5.is_file():
        with h5py.File(str(inc_h5), "r") as hf:
            inc = np.asarray(hf["inc"]).ravel()
        inc_mean = float(np.mean(inc[inc != 0]))
        logger.info("Loaded incidence angle from %s (mean=%.4f rad)", inc_h5.name, inc_mean)
        return inc_mean

    # Try inc1.mat
    inc_mat = patch_dir / "inc1.mat"
    if inc_mat.is_file():
        d = load_mat(inc_mat)
        inc = np.asarray(d["inc"]).ravel().astype(np.float64)
        inc_mean = float(np.mean(inc[inc != 0]))
        logger.info("Loaded incidence angle from %s (mean=%.4f rad)", inc_mat.name, inc_mean)
        return inc_mean

    # Try la1.h5
    la_h5 = patch_dir / "la1.h5"
    if la_h5.is_file():
        with h5py.File(str(la_h5), "r") as hf:
            la = np.asarray(hf["la"]).ravel()
        inc_mean = float(np.mean(la)) + 0.052
        logger.info("Loaded look angle from %s, inc_mean=%.4f rad", la_h5.name, inc_mean)
        return inc_mean

    # Try la1.mat
    la_mat = patch_dir / "la1.mat"
    if la_mat.is_file():
        d = load_mat(la_mat)
        la = np.asarray(d["la"]).ravel().astype(np.float64)
        inc_mean = float(np.mean(la)) + 0.052
        logger.info("Loaded look angle from %s, inc_mean=%.4f rad", la_mat.name, inc_mean)
        return inc_mean

    # Default — MATLAB line 131: inc_mean = 21*pi/180
    inc_mean = 21.0 * np.pi / 180.0
    logger.warning("No inc/la file found — using default inc_mean=%.4f rad (21°)", inc_mean)
    return inc_mean


# ============================================================================
# HDF5 persistence for pm1
# ============================================================================

def _save_pm_h5(
    h5_path: Path,
    *,
    ph_patch: np.ndarray,
    K_ps: np.ndarray,
    C_ps: np.ndarray,
    coh_ps: np.ndarray,
    N_opt: np.ndarray,
    ph_res: np.ndarray,
    step_number: int,
    ph_grid: np.ndarray,
    n_trial_wraps: float,
    grid_ij: np.ndarray,
    grid_size: float,
    low_pass: np.ndarray,
    i_loop: int,
    ph_weight: np.ndarray,
    Nr: np.ndarray,
    Nr_max_nz_ix: int,
    coh_bins: np.ndarray,
    coh_ps_save: np.ndarray,
    gamma_change_save: float,
    weighting: np.ndarray,
    save_restart_state: bool = False,
) -> None:
    """Persist Step-2 outputs to ``pm1.h5``.

    Dataset layout mirrors MATLAB ``stamps_save(pmname, …)``
    (ps_est_gamma_quick.m line 322).

    For MATLAB compatibility the arrays are stored with the same dtypes
    as the original ``.mat`` file:
      * ``ph_patch``, ``ph_grid``, ``ph_weight`` → complex64
      * ``ph_res`` → float32
      * ``K_ps``, ``C_ps``, ``coh_ps`` → float64
      * ``grid_ij`` → float32 (1-based indices, matching MATLAB)
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing Step-2 HDF5 output: %s", h5_path)

    logger.info(
        "Step-2 save options: restart_state=%s",
        "yes" if save_restart_state else "no",
    )

    with h5py.File(str(h5_path), "w") as hf:
        # Large arrays with compression
        hf.create_dataset("ph_patch", data=np.ascontiguousarray(ph_patch, dtype=np.complex64), **_FAST_COMPRESSION)
        hf.create_dataset("ph_res", data=np.ascontiguousarray(ph_res, dtype=np.float32), **_FAST_COMPRESSION)
        hf.create_dataset("ph_grid", data=np.ascontiguousarray(ph_grid, dtype=np.complex64), **_FAST_COMPRESSION)
        if save_restart_state:
            hf.create_dataset("ph_weight", data=np.ascontiguousarray(ph_weight, dtype=np.complex64), **_FAST_COMPRESSION)
        hf.attrs["ph_weight_saved"] = bool(save_restart_state)

        # Per-pixel vectors
        hf.create_dataset("K_ps", data=K_ps.astype(np.float64))
        hf.create_dataset("C_ps", data=C_ps.astype(np.float64))
        hf.create_dataset("coh_ps", data=coh_ps.astype(np.float64))
        hf.create_dataset("N_opt", data=N_opt.astype(np.uint8))
        if save_restart_state:
            hf.create_dataset("coh_ps_save", data=coh_ps_save.astype(np.float64))
            hf.create_dataset("weighting", data=weighting.astype(np.float64))

        # Grid metadata
        hf.create_dataset("grid_ij", data=grid_ij.astype(np.float32))
        hf.create_dataset("grid_size", data=np.float64(grid_size))

        # Filter / convergence metadata
        hf.create_dataset("low_pass", data=low_pass.astype(np.float64))
        hf.create_dataset("n_trial_wraps", data=np.float64(n_trial_wraps))
        hf.create_dataset("Nr", data=Nr.astype(np.float64))
        hf.create_dataset("Nr_max_nz_ix", data=np.int32(Nr_max_nz_ix))
        hf.create_dataset("coh_bins", data=coh_bins.astype(np.float64))

        # Scalars
        hf.create_dataset("step_number", data=np.int32(step_number))
        hf.create_dataset("i_loop", data=np.int32(i_loop))
        hf.create_dataset("gamma_change_save", data=np.float64(gamma_change_save))

    sz = h5_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", h5_path.name, sz)


# ============================================================================
# Main estimator class
# ============================================================================

class GammaEstimator:
    """Estimate PS coherence (gamma) — port of ``ps_est_gamma_quick.m``.

    Parameters
    ----------
    patch_dir : Path
        Directory containing ``ps1.h5`` (or ``.mat`` equivalents).
    psver : int
        PS version number (default 1).
    """

    # MATLAB constant: rho = 830000 (mean range in metres)
    RHO: float = 830_000.0
    # MATLAB constant: n_rand = 300000 (random samples)
    N_RAND: int = 300_000
    # Chunk size for random-phase simulation to limit memory
    RAND_CHUNK: int = 30_000

    def __init__(self, patch_dir: Union[str, Path], psver: int = 1) -> None:
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver

        # Config — the singleton may already be initialised by StampsRunner.
        # If not (standalone run), initialise with patch_dir so that parms.mat
        # can be found in patch_dir or its parent (matching MATLAB search order).
        self._cfg = StampsConfig(work_dir=self.patch_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, restart_flag: int = 0) -> None:
        """Execute the gamma estimation.

        Parameters
        ----------
        restart_flag : int
            0 = fresh start, 1 = restart from saved state,
            2 = restart but only recalculate patch values.
        """
        t_start = time.time()
        logger.info("Estimating gamma for candidate pixels")

        # --- Load parameters ---
        parms = self._load_parameters()

        # --- Build Butterworth low-pass ---
        low_pass = _butterworth_lowpass(
            int(parms["clap_win"]),
            parms["grid_size"],
            parms["low_pass_wavelength"],
        )

        # --- Load data from ps1.h5 / mat files ---
        (ph, bperp_1d, bperp_mat, xy, D_A,
         n_ps, n_ifg, n_image,
         master_ix, ifgday_ix,
         small_baseline_flag) = self._load_ps_data(parms)

        # --- Handle SB vs PS mode (lines 94-107) ---
        # MATLAB: for PS mode, remove master column from ph and bperp
        if small_baseline_flag == "y":
            # SB mode: keep everything as-is
            logger.info("SB mode: n_ifg=%d, n_image=%d", n_ifg, n_image)
        else:
            # PS mode: remove master column from ph (and from bperp_1d/bperp_mat only if they have n_image size)
            # Step 1 ps1.h5 stores: ph (n_ps, n_image), bperp (n_image,), bperp_mat (n_ps, n_ifg) — bperp_mat already excludes master
            # master_ix in h5 is 1-based (MATLAB convention)
            midx = int(master_ix) - 1  # convert to 0-based
            n_col_ph = ph.shape[1]
            keep = np.concatenate([np.arange(midx), np.arange(midx + 1, n_col_ph)])
            ph = ph[:, keep]
            if bperp_1d.size == n_col_ph:
                bperp_1d = bperp_1d[keep]
            if bperp_mat.shape[1] == n_col_ph:
                bperp_mat = bperp_mat[:, keep]
            n_ifg = ph.shape[1]
            logger.info("PS mode: removed master column → n_ifg=%d", n_ifg)

        # --- Normalise phase (lines 110-113) ---
        # MATLAB: A = abs(ph); A(A==0)=1; ph = ph./A
        A = np.abs(ph).astype(np.float32)
        A[A == 0] = 1.0
        ph = (ph / A).astype(np.complex64)

        # Identify null pixels (any zero in any ifg → MATLAB lines 89-92)
        # MATLAB: null_i = unique rows with any zero
        has_zero = np.any(ph == 0, axis=1)
        good_ix = ~has_zero
        logger.info("Pixels with all non-zero phase: %d / %d", int(good_ix.sum()), n_ps)

        # --- Incidence angle & max_K (lines 118-133) ---
        inc_mean = _load_inc_mean(self.patch_dir)
        max_K = parms["max_topo_err"] / (
            parms["lambda"] * self.RHO * np.sin(inc_mean) / (4 * np.pi)
        )

        bp_range_global = float(bperp_1d.max() - bperp_1d.min())
        n_trial_wraps = bp_range_global * max_K / (2 * np.pi)
        logger.info("n_trial_wraps = %.6f", n_trial_wraps)

        # --- Determine low coherence threshold (lines 48-52) ---
        if small_baseline_flag == "y":
            low_coh_thresh = 15
        else:
            low_coh_thresh = 31

        # ================================================================
        # Initialisation (restart_flag == 0 path, lines 153-198)
        # ================================================================
        if restart_flag > 0:
            logger.info("Restarting from saved state (restart_flag=%d)", restart_flag)
            # Load previous pm state — TODO: implement restart loading
            raise NotImplementedError("Restart is not yet implemented in Python port.")
        else:
            logger.info("Initialising random distribution...")
            rng = np.random.RandomState(2005)

            # --- Random coherence distribution (lines 158-171) ---
            Nr, coh_bins, Nr_max_nz_ix = self._compute_random_distribution(
                rng, n_ifg, n_image, bperp_1d, n_trial_wraps,
                small_baseline_flag, ifgday_ix,
            )

            # --- Initialise working arrays (lines 182-197) ---
            K_ps = np.zeros(n_ps, dtype=np.float64)
            C_ps = np.zeros(n_ps, dtype=np.float64)
            coh_ps = np.zeros(n_ps, dtype=np.float64)
            coh_ps_save = np.zeros(n_ps, dtype=np.float64)
            N_opt = np.zeros(n_ps, dtype=np.int32)
            ph_res = np.zeros((n_ps, n_ifg), dtype=np.float32)
            ph_patch = np.zeros_like(ph, dtype=np.complex64)

            # --- Grid indices (lines 190-193) ---
            # MATLAB: grid_ij(:,1) = ceil((xy(:,3) - min(xy(:,3)) + 1e-6) / grid_size)
            #         grid_ij(:,2) = ceil((xy(:,2) - min(xy(:,2)) + 1e-6) / grid_size)
            grid_size = parms["grid_size"]
            grid_ij_1 = np.ceil((xy[:, 2] - xy[:, 2].min() + 1e-6) / grid_size).astype(np.int32)
            max_i = grid_ij_1.max()
            grid_ij_1[grid_ij_1 == max_i] = max_i - 1

            grid_ij_2 = np.ceil((xy[:, 1] - xy[:, 1].min() + 1e-6) / grid_size).astype(np.int32)
            max_j = grid_ij_2.max()
            grid_ij_2[grid_ij_2 == max_j] = max_j - 1

            # 1-based grid indices for saving (matching MATLAB pm1.mat)
            grid_ij_save = np.column_stack([grid_ij_1, grid_ij_2]).astype(np.float32)
            # 0-based for internal array indexing
            grid_ij_0 = np.column_stack([grid_ij_1 - 1, grid_ij_2 - 1])

            i_loop = 1
            weighting = 1.0 / D_A
            gamma_change_save = 0.0

        # --- Grid dimensions ---
        n_i = int(grid_ij_1.max())   # MATLAB: n_i = max(grid_ij(:,1))
        n_j = int(grid_ij_2.max())

        logger.info("%d PS candidates to process (grid %d × %d)", n_ps, n_i, n_j)

        # MATLAB line 206: xy(:,1) = 1:n_ps (assume already sorted)
        # In our data xy[:,0] is the ID column — not modified for computation.

        loop_end = False

        # ================================================================
        # Main iterative loop (lines 210-324)
        # ================================================================
        while not loop_end:
            logger.info("Iteration #%d", i_loop)

            # --- Step 1: Calculate patch phases (lines 215-230) ---
            logger.info("Calculating patch phases...")
            t_patch = time.time()

            # MATLAB line 217:
            #   ph_weight = ph .* exp(-j*bp.bperp_mat .* repmat(K_ps,1,n_ifg))
            #               .* repmat(weighting, 1, n_ifg)
            ph_weight = (
                ph
                * np.exp(-1j * bperp_mat * K_ps[:, None])
                * weighting[:, None]
            ).astype(np.complex64)

            # Grid accumulation (vectorised — replaces MATLAB for-loop lines 219-222)
            # MATLAB: ph_grid(grid_ij(i,1), grid_ij(i,2), :) += ph_weight(i,:)
            ph_grid = np.zeros((n_i, n_j, n_ifg), dtype=np.complex64)
            linear_idx = grid_ij_0[:, 0] * n_j + grid_ij_0[:, 1]
            ph_grid_flat = ph_grid.reshape(-1, n_ifg)
            np.add.at(ph_grid_flat, linear_idx, ph_weight)
            ph_grid = ph_grid_flat.reshape(n_i, n_j, n_ifg)

            # --- CLAP filtering (lines 224-226) ---
            # MATLAB: ph_filt(:,:,i) = clap_filt(ph_grid(:,:,i), ...)
            ph_filt = np.zeros_like(ph_grid)
            clap_n_win = int(parms["clap_win"] * 0.75)
            clap_n_pad = int(parms["clap_win"] * 0.25)
            for k in range(n_ifg):
                ph_filt[:, :, k] = clap_filt(
                    ph_grid[:, :, k],
                    alpha=parms["clap_alpha"],
                    beta=parms["clap_beta"],
                    n_win=clap_n_win,
                    n_pad=clap_n_pad,
                    low_pass=low_pass,
                )

            # Extract filtered phase per pixel (vectorised — replaces lines 228-230)
            # MATLAB: ph_patch(i, 1:n_ifg) = squeeze(ph_filt(grid_ij(i,1), grid_ij(i,2), :))
            ph_patch = ph_filt[grid_ij_0[:, 0], grid_ij_0[:, 1], :]  # (n_ps, n_ifg)
            ph_patch = ph_patch.astype(np.complex64)

            # Normalise ph_patch (lines 233-234)
            # MATLAB: ix = ph_patch~=0; ph_patch(ix) = ph_patch(ix) ./ abs(ph_patch(ix))
            nz = ph_patch != 0
            ph_patch[nz] = ph_patch[nz] / np.abs(ph_patch[nz])

            logger.info("Patch phase computation: %.1f s", time.time() - t_patch)

            # ---- Step 2: Estimate topo error (lines 239-260) ----
            if restart_flag < 2:
                logger.info("Estimating topo error...")
                t_topo = time.time()

                # Differential phase: psdph = ph .* conj(ph_patch)
                # MATLAB line 245: psdph = ph(i,:) .* conj(ph_patch(i,:))
                psdph = ph * np.conj(ph_patch)  # (n_ps, n_ifg)

                # Batch ps_topofit — replaces the per-pixel for-loop entirely
                K_ps, C_ps, coh_ps, ph_res, N_opt = ps_topofit_batch(
                    psdph, bperp_mat, n_trial_wraps
                )

                logger.info("Topo error estimation: %.1f s", time.time() - t_topo)

                # --- Convergence check (lines 273-282) ---
                gamma_change_rms = float(
                    np.sqrt(np.sum((coh_ps - coh_ps_save) ** 2) / n_ps)
                )
                gamma_change_change = gamma_change_rms - gamma_change_save
                logger.info("gamma_change_change = %.6f", gamma_change_change)
                gamma_change_save = gamma_change_rms
                coh_ps_save = coh_ps.copy()

                # Re-read convergence thresholds (they may have been updated)
                gamma_conv = float(self._cfg.getparm("gamma_change_convergence"))
                gamma_max_iter = int(self._cfg.getparm("gamma_max_iterations"))

                if (abs(gamma_change_change) < gamma_conv) or (i_loop >= gamma_max_iter):
                    loop_end = True
                    logger.info(
                        "Converged (|Δγ|=%.6f < %.6f or iter=%d ≥ %d)",
                        abs(gamma_change_change), gamma_conv,
                        i_loop, gamma_max_iter,
                    )
                else:
                    i_loop += 1
                    weighting, Nr = self._update_weighting(
                        parms["filter_weighting"],
                        coh_ps, coh_bins, Nr, low_coh_thresh, Nr_max_nz_ix,
                        A, ph_res, n_ifg,
                    )
            else:
                loop_end = True

            # Save only the final iteration. Restart loading is not implemented
            # in this port, so intermediate pm1.h5 writes only add substantial
            # I/O time without enabling recovery.
            if loop_end:
                pm_path = self.patch_dir / f"pm{self.psver}.h5"
                _save_pm_h5(
                    pm_path,
                    ph_patch=ph_patch,
                    K_ps=K_ps,
                    C_ps=C_ps,
                    coh_ps=coh_ps,
                    N_opt=N_opt,
                    ph_res=ph_res,
                    step_number=1,    # MATLAB resets to 1 after topo-fit iteration
                    ph_grid=ph_grid,
                    n_trial_wraps=n_trial_wraps,
                    grid_ij=grid_ij_save,
                    grid_size=parms["grid_size"],
                    low_pass=low_pass,
                    i_loop=i_loop,
                    ph_weight=ph_weight,
                    Nr=Nr,
                    Nr_max_nz_ix=Nr_max_nz_ix,
                    coh_bins=coh_bins,
                    coh_ps_save=coh_ps_save,
                    gamma_change_save=gamma_change_save,
                    weighting=weighting,
                    save_restart_state=_env_flag("STAMPS_SAVE_PM_RESTART_STATE", False),
                )
            else:
                logger.info("Skipping intermediate pm1.h5 save (restart loading is not implemented)")

        elapsed = time.time() - t_start
        logger.info("Step 2 (gamma estimation) completed in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Internal: load parameters from StampsConfig
    # ------------------------------------------------------------------
    def _load_parameters(self) -> dict:
        """Read all parameters needed by ps_est_gamma_quick from config.

        MATLAB correspondence (lines 36-46)::

            grid_size             = getparm('filter_grid_size', 1)
            filter_weighting      = getparm('filter_weighting', 1)
            n_win                 = getparm('clap_win', 1)
            low_pass_wavelength   = getparm('clap_low_pass_wavelength', 1)
            clap_alpha            = getparm('clap_alpha', 1)
            clap_beta             = getparm('clap_beta', 1)
            max_topo_err          = getparm('max_topo_err', 1)
            lambda                = getparm('lambda', 1)
            gamma_change_convergence = getparm('gamma_change_convergence', 1)
            gamma_max_iterations  = getparm('gamma_max_iterations', 1)
            small_baseline_flag   = getparm('small_baseline_flag', 1)
        """
        cfg = self._cfg
        return {
            "grid_size": float(cfg.getparm("filter_grid_size")),
            "filter_weighting": str(cfg.getparm("filter_weighting")),
            "clap_win": float(cfg.getparm("clap_win")),
            "low_pass_wavelength": float(cfg.getparm("clap_low_pass_wavelength")),
            "clap_alpha": float(cfg.getparm("clap_alpha")),
            "clap_beta": float(cfg.getparm("clap_beta")),
            "max_topo_err": float(cfg.getparm("max_topo_err")),
            "lambda": float(cfg.getparm("lambda")),
            "gamma_change_convergence": float(cfg.getparm("gamma_change_convergence")),
            "gamma_max_iterations": int(cfg.getparm("gamma_max_iterations")),
            "small_baseline_flag": str(cfg.getparm("small_baseline_flag")).strip().lower(),
        }

    # ------------------------------------------------------------------
    # Internal: load PS data from HDF5 (ps1.h5) or .mat files
    # ------------------------------------------------------------------
    def _load_ps_data(self, parms: dict):
        """Load Step-1 output data.

        MATLAB correspondence (lines 60-80)::

            ps  = load(psname)   → ps1.mat  (n_ps, n_ifg, bperp, xy, …)
            bp  = load(bpname)   → bp1.mat  (bperp_mat)
            da  = load(daname)   → da1.mat  (D_A)
            ph  = load(phname)   → ph1.mat  (ph)   [overrides ps.ph]

        In Python these are consolidated in ``ps1.h5`` from Step 1.
        We also check for a separate ``ph1.h5`` override.

        Returns a tuple of arrays and scalars.
        """
        ps_h5 = self.patch_dir / f"ps{self.psver}.h5"
        if not ps_h5.is_file():
            raise FileNotFoundError(f"Step-1 output not found: {ps_h5}")

        logger.info("Loading Step-1 data from %s", ps_h5)

        with h5py.File(str(ps_h5), "r") as hf:
            # Scalars
            n_ps = int(hf["n_ps"][()])
            n_ifg = int(hf["n_ifg"][()])
            n_image = int(hf["n_image"][()])
            master_ix = int(hf["master_ix"][()])  # 1-based

            # Arrays — use slicing to enable future chunk-wise reading
            bperp_1d = np.asarray(hf["bperp"][:], dtype=np.float64)
            xy = np.asarray(hf["xy"][:], dtype=np.float32)

            # bperp_mat: (n_ps, n_ifg) float32 — may be large
            bperp_mat = np.asarray(hf["bperp_mat"][:], dtype=np.float32)

            # D_A — optional, default to ones
            if "D_A" in hf:
                D_A = np.asarray(hf["D_A"][:], dtype=np.float64).ravel()
            else:
                D_A = np.ones(n_ps, dtype=np.float64)

            # ifgday_ix — needed for SB mode
            if "ifgday_ix" in hf:
                ifgday_ix = np.asarray(hf["ifgday_ix"][:], dtype=np.int32)
            else:
                ifgday_ix = None

            # ph — check for separate ph file first
            ph = self._load_ph(hf, n_ps, n_ifg)

        small_baseline_flag = parms["small_baseline_flag"]

        return (ph, bperp_1d, bperp_mat, xy, D_A,
                n_ps, n_ifg, n_image,
                master_ix, ifgday_ix,
                small_baseline_flag)

    def _load_ph(self, hf: h5py.File, n_ps: int, n_ifg: int) -> np.ndarray:
        """Load complex phase. Prefer Step-1 .h5 output (ph1.h5 or ps1.h5), then .mat.

        MATLAB correspondence (lines 81-87)::

            if exist([phname,'.mat'], 'file')
                phin = load(phname);  ph = phin.ph;
            else
                ph = ps.ph;
            end
        """
        # 1) Prefer ph inside already-open ps1.h5 (Step 1 Python output)
        if "ph" in hf:
            logger.info("Loading ph from ps1.h5 /ph dataset")
            return np.asarray(hf["ph"][:])

        # 2) Standalone ph1.h5
        ph_h5 = self.patch_dir / f"ph{self.psver}.h5"
        if ph_h5.is_file():
            logger.info("Loading ph from standalone %s", ph_h5.name)
            with h5py.File(str(ph_h5), "r") as phf:
                return np.asarray(phf["ph"][:])

        # 3) Fall back to ph1.mat only if no .h5 source
        ph_mat = self.patch_dir / f"ph{self.psver}.mat"
        if ph_mat.is_file():
            logger.info("Loading ph from %s", ph_mat.name)
            d = load_mat(ph_mat)
            return np.asarray(d["ph"])

        raise FileNotFoundError(
            f"Complex phase data not found in {self.patch_dir} "
            f"(checked ph{self.psver}.h5, ph{self.psver}.mat, ps{self.psver}.h5:/ph)"
        )

    # ------------------------------------------------------------------
    # Internal: random coherence distribution
    # ------------------------------------------------------------------
    def _compute_random_distribution(
        self,
        rng: np.random.RandomState,
        n_ifg: int,
        n_image: int,
        bperp_1d: np.ndarray,
        n_trial_wraps: float,
        small_baseline_flag: str,
        ifgday_ix: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Simulate random-phase pixels to determine the coherence PDF.

        MATLAB correspondence (lines 154-179)::

            rand('state', 2005)
            if strcmpi(small_baseline_flag, 'y')
                rand_image = 2*pi*rand(n_rand, n_image);
                rand_ifg = rand_image(:, ifgday_ix(:,2)) - …(:, ifgday_ix(:,1));
            else
                rand_ifg = 2*pi*rand(n_rand, n_ifg);
            end
            for i = n_rand:-1:1
                [~, ~, coh_r] = ps_topofit(exp(j*rand_ifg(i,:)), bperp, …);
                coh_rand(i) = coh_r(1);
            end
            coh_bins = 0.005:0.01:0.995;
            Nr = hist(coh_rand, coh_bins);

        The per-sample ``ps_topofit`` loop is replaced by the vectorised
        ``_ps_topofit_rand_batch`` (processes chunks of random samples at once).

        Returns
        -------
        Nr : (100,) float64 — histogram counts of random coherence
        coh_bins : (100,) float64 — bin centres
        Nr_max_nz_ix : int — index of last non-zero Nr bin (0-based)
        """
        n_rand = self.N_RAND
        chunk = self.RAND_CHUNK

        coh_rand = np.empty(n_rand, dtype=np.float64)

        # Process in chunks to limit memory usage
        for c_start in range(0, n_rand, chunk):
            c_end = min(c_start + chunk, n_rand)
            c_size = c_end - c_start

            if small_baseline_flag == "y" and ifgday_ix is not None:
                # SB mode: random images → random interferograms via differencing
                # MATLAB: rand_image = 2*pi*rand(n_rand, n_image)
                # ifgday_ix in h5 is 1-based; convert to 0-based for indexing
                ifg_ix = ifgday_ix - 1  # (n_ifg, 2), 0-based
                rand_image = 2 * np.pi * rng.rand(c_size, n_image)
                rand_ifg = rand_image[:, ifg_ix[:, 1]] - rand_image[:, ifg_ix[:, 0]]
                del rand_image
            else:
                rand_ifg = 2 * np.pi * rng.rand(c_size, n_ifg)

            rand_cpx = np.exp(1j * rand_ifg)
            del rand_ifg

            coh_rand[c_start:c_end] = _ps_topofit_rand_batch(
                rand_cpx, bperp_1d, n_trial_wraps,
            )
            del rand_cpx

            if c_end % (chunk * 3) == 0 or c_end == n_rand:
                logger.info(
                    "Random coherence: %d / %d samples processed", c_end, n_rand
                )

        # Histogram — MATLAB: coh_bins = 0.005:0.01:0.995; Nr = hist(…)
        coh_bins = np.arange(0.005, 1.0, 0.01)  # 100 bins
        Nr, _ = np.histogram(coh_rand, bins=np.arange(0.0, 1.01, 0.01))
        Nr = Nr.astype(np.float64)

        # Find last non-zero bin — MATLAB lines 176-179
        nz_indices = np.nonzero(Nr)[0]
        Nr_max_nz_ix = int(nz_indices[-1]) if len(nz_indices) > 0 else 0

        logger.info("Random distribution: Nr_max_nz_ix=%d", Nr_max_nz_ix)
        return Nr, coh_bins, Nr_max_nz_ix

    # ------------------------------------------------------------------
    # Internal: weight update
    # ------------------------------------------------------------------
    def _update_weighting(
        self,
        filter_weighting: str,
        coh_ps: np.ndarray,
        coh_bins: np.ndarray,
        Nr: np.ndarray,
        low_coh_thresh: int,
        Nr_max_nz_ix: int,
        A: np.ndarray,
        ph_res: np.ndarray,
        n_ifg: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update pixel weighting for the next iteration.

        Returns (weighting, Nr) — Nr is modified in-place when P-square
        is used, matching MATLAB's cumulative scaling behaviour.

        MATLAB correspondence (lines 291-315)::

            if strcmpi(filter_weighting, 'P-square')
                Na = hist(coh_ps, coh_bins);
                Nr = Nr * sum(Na(1:low_coh_thresh)) / sum(Nr(1:low_coh_thresh));
                ...
                weighting = (1 - Prand_ps).^2;
            else
                g = mean(A .* cos(ph_res), 2);
                sigma_n = sqrt(0.5 * (mean(A.^2, 2) - g.^2));
                weighting = g ./ sigma_n;   % SNR
            end
        """
        if filter_weighting.strip().lower() == "p-square":
            w, Nr_out = self._weighting_p_square(
                coh_ps, coh_bins, Nr, low_coh_thresh, Nr_max_nz_ix,
            )
            return w, Nr_out
        else:
            return self._weighting_snr(A, ph_res, n_ifg), Nr

    def _weighting_p_square(
        self,
        coh_ps: np.ndarray,
        coh_bins: np.ndarray,
        Nr: np.ndarray,
        low_coh_thresh: int,
        Nr_max_nz_ix: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """P-square weighting strategy.

        Returns (weighting, Nr_scaled).  Nr is scaled **in-place** just as
        in MATLAB, where ``Nr = Nr * sum(Na…)/sum(Nr…)`` accumulates across
        iterations.

        MATLAB correspondence (lines 292-304)::

            Na = hist(coh_ps, coh_bins);
            Nr = Nr * sum(Na(1:low_coh_thresh)) / sum(Nr(1:low_coh_thresh));
            Na(Na==0) = 1;
            Prand = Nr ./ Na;
            Prand(1:low_coh_thresh) = 1;
            Prand(Nr_max_nz_ix+1:end) = 0;
            Prand(Prand>1) = 1;
            Prand = filter(gausswin(7), 1, [ones(1,7), Prand]) / sum(gausswin(7));
            Prand = Prand(8:end);
            Prand = interp([1, Prand], 10);
            Prand = Prand(1:end-9);
            Prand_ps = Prand(round(coh_ps*1000) + 1)';
            weighting = (1 - Prand_ps).^2;
        """
        from scipy.signal import lfilter, resample_poly

        # Na: histogram of actual coherence — MATLAB: Na = hist(coh_ps, coh_bins)
        Na, _ = np.histogram(coh_ps, bins=np.arange(0.0, 1.01, 0.01))
        Na = Na.astype(np.float64)

        # Scale random distribution to match actual low-coherence counts
        # MATLAB: Nr = Nr * sum(Na(1:low_coh_thresh)) / sum(Nr(1:low_coh_thresh))
        # This modifies Nr IN-PLACE across iterations (cumulative scaling)
        sum_Na_low = Na[:low_coh_thresh].sum()
        sum_Nr_low = Nr[:low_coh_thresh].sum()
        if sum_Nr_low > 0:
            Nr *= (sum_Na_low / sum_Nr_low)

        Na_safe = Na.copy()
        Na_safe[Na_safe == 0] = 1.0

        Prand = Nr / Na_safe
        Prand[:low_coh_thresh] = 1.0
        Prand[Nr_max_nz_ix + 1:] = 0.0
        Prand[Prand > 1] = 1.0

        # Smooth with gausswin(7) FIR filter
        # MATLAB: Prand = filter(gausswin(7), 1, [ones(1,7), Prand]) / sum(gausswin(7))
        gw = _gausswin(7)
        padded = np.concatenate([np.ones(7), Prand])
        filtered = lfilter(gw, 1.0, padded) / gw.sum()
        Prand = filtered[7:]  # remove warmup transient

        # Interpolate to 1000 bins (0.001 resolution)
        # MATLAB: Prand = interp([1, Prand], 10);  Prand = Prand(1:end-9)
        Prand_ext = np.concatenate([[1.0], Prand])  # length 101
        Prand_up = resample_poly(Prand_ext, 10, 1)  # length 1010
        Prand_fine = Prand_up[:len(Prand_up) - 9]   # length 1001

        # Map pixel coherence to Prand — MATLAB: Prand_ps = Prand(round(coh_ps*1000)+1)
        idx = np.round(coh_ps * 1000).astype(np.int64)
        idx = np.clip(idx, 0, len(Prand_fine) - 1)
        Prand_ps = Prand_fine[idx]

        weighting = (1.0 - Prand_ps) ** 2
        return weighting, Nr

    def _weighting_snr(
        self,
        A: np.ndarray,
        ph_res: np.ndarray,
        n_ifg: int,
    ) -> np.ndarray:
        """SNR-based weighting (non P-square branch).

        MATLAB correspondence (lines 309-314)::

            g = mean(A .* cos(ph_res), 2);
            sigma_n = sqrt(0.5 * (mean(A.^2, 2) - g.^2));
            weighting(sigma_n==0) = 0;
            weighting(sigma_n~=0) = g(sigma_n~=0) ./ sigma_n(sigma_n~=0);
        """
        g = np.mean(A * np.cos(ph_res), axis=1)
        sigma_n = np.sqrt(np.maximum(0.5 * (np.mean(A ** 2, axis=1) - g ** 2), 0.0))

        weighting = np.zeros_like(g)
        nz = sigma_n != 0
        weighting[nz] = g[nz] / sigma_n[nz]
        return weighting

    # ------------------------------------------------------------------
    # Internal: load parameter 'lambda' from ps1.h5 or config
    # ------------------------------------------------------------------
    def _get_lambda(self) -> float:
        """Return wavelength lambda (in metres)."""
        val = self._cfg.getparm("lambda")
        if val is not None:
            return float(val)
        # Fallback: C-band default
        return 0.0556


# ============================================================================
# Module-level convenience function
# ============================================================================

def ps_est_gamma_quick(
    patch_dir: Union[str, Path],
    restart_flag: int = 0,
    psver: int = 1,
) -> None:
    """Module-level entry point (mirrors MATLAB ``ps_est_gamma_quick``).

    Parameters
    ----------
    patch_dir : str or Path
        Patch directory containing ``ps1.h5`` and auxiliary files.
    restart_flag : int
        0 = fresh start, 1 = restart, 2 = recalc patch only.
    psver : int
        PS version number.
    """
    estimator = GammaEstimator(patch_dir=patch_dir, psver=psver)
    estimator.run(restart_flag=restart_flag)


# ============================================================================
# CLI demo
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 2: Estimate gamma (ps_est_gamma_quick)"
    )
    parser.add_argument(
        "patch_dir", type=str,
        help="Path to patch directory containing ps1.h5",
    )
    parser.add_argument(
        "--restart", type=int, default=0,
        help="Restart flag (0=fresh, 1=restart, 2=recalc patch)",
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

    ps_est_gamma_quick(args.patch_dir, restart_flag=args.restart, psver=args.psver)
