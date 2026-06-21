"""
uw_core.py — uw_3d internal chain: grid → interp → space_time → stat_costs → from_grid
===============================================================================

Implements MATLAB uw_grid_wrapped, uw_interp, uw_sb_unwrap_space_time,
uw_stat_costs, uw_unwrap_from_grid. Used by phase_unwrapping.UnwrapPipeline._run_uw_3d.

Snaphu path: from getparm('snaphu_path') or env SNAPHU_PATH or default Windows path.

Cygwin build note: If snaphu prints "cygwin_exception::open_stackdumpfile" and crashes,
the binary is likely a Cygwin build and can be unstable on Windows. Use a non-Cygwin
Windows build instead, e.g.:
  - https://github.com/marsfan/SNAPHU-win (pre-built Windows binaries)
  - ESA STEP snaphu-v1.4.2_win64 (MSYS2 build), or run Snaphu under WSL/Linux.
Set SNAPHU_PATH or parm snaphu_path to the path of the replacement executable.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree

logger = logging.getLogger("stamps")

# Default snaphu executable (Windows); override via SNAPHU_PATH or getparm('snaphu_path')
DEFAULT_SNAPHU_PATH = r"D:\env\snaphu-v1.4.2_win64\bin\snaphu.exe"


def get_snaphu_path(cfg: Optional[Any] = None) -> str:
    """Return path to snaphu executable.
    Precedence: env SNAPHU_PATH or snaphu_path, then getparm('snaphu_path'), then default.
    If the value is a directory, appends snaphu.exe (Windows) or snaphu (else).
    """
    raw = (
        os.environ.get("SNAPHU_PATH")
        or os.environ.get("snaphu_path")
    )
    if not raw and cfg is not None:
        try:
            p = cfg.getparm("snaphu_path")
            if p is not None and str(p).strip():
                raw = str(p).strip()
        except Exception:
            pass
    if not raw:
        raw = DEFAULT_SNAPHU_PATH
    raw = str(raw).strip()
    path = Path(raw)
    if path.is_dir():
        exe = "snaphu.exe" if os.name == "nt" else "snaphu"
        path = path / exe
    return str(path)


# ---------------------------------------------------------------------------
# wrap_filt — Goldstein + optional lowpass (MATLAB wrap_filt.m)
# ---------------------------------------------------------------------------

def wrap_filt(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    low_flag: str = "n",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Goldstein adaptive filter; optional lowpass. Returns (ph_out, ph_out_low or None)."""
    n_i, n_j = ph.shape
    n_pad = max(1, int(round(n_win * 0.25)))
    n_inc = max(1, n_win // 2)
    n_win_i = max(0, (n_i + n_inc - 1) // n_inc - 1)
    n_win_j = max(0, (n_j + n_inc - 1) // n_inc - 1)
    ph = np.nan_to_num(ph, nan=0.0, copy=True)
    ph_out = np.zeros_like(ph, dtype=np.complex128)
    ph_out_low = np.zeros_like(ph, dtype=np.complex128) if low_flag == "y" else None

    # MATLAB gausswin(N, alpha) has std=(N-1)/(2*alpha).
    from scipy.signal.windows import gaussian as gaussian_win

    smooth_win = gaussian_win(7, (7 - 1) / (2 * 2.5))
    B = np.outer(smooth_win, smooth_win)
    low_std = (n_win + n_pad - 1) / (2 * 16)
    low_win = gaussian_win(n_win + n_pad, low_std)
    L = np.fft.ifftshift(np.outer(low_win, low_win))

    x = np.arange(1, n_win // 2 + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    wind_func = X + Y
    wind_func = np.hstack([wind_func, np.fliplr(wind_func)])
    wind_func = np.vstack([wind_func, np.flipud(wind_func)])

    from scipy.signal import convolve2d

    for ix1 in range(n_win_i):
        wf = wind_func
        i1 = ix1 * n_inc
        i2 = i1 + n_win
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack([np.zeros((i_shift, n_win)), wf[: n_win - i_shift, :]])
        for ix2 in range(n_win_j):
            wf2 = wf
            j1 = ix2 * n_inc
            j2 = j1 + n_win
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.hstack([np.zeros((n_win, j_shift)), wf2[:, : n_win - j_shift]])

            ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=np.complex128)
            ph_bit[:n_win, :n_win] = ph[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = np.fft.ifftshift(
                convolve2d(np.fft.fftshift(H), B, mode="same", boundary="fill", fillvalue=0)
            )
            meanH = np.median(H)
            if meanH != 0:
                H = H / meanH
            H = np.power(H, alpha)

            ph_filt = np.fft.ifft2(ph_fft * H)
            ph_out[i1:i2, j1:j2] += ph_filt[:n_win, :n_win] * wf2
            if low_flag == "y" and ph_out_low is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * L)
                ph_out_low[i1:i2, j1:j2] += ph_filt_low[:n_win, :n_win] * wf2

    mag = np.abs(ph)
    ph_out = mag * np.exp(1j * np.angle(ph_out))
    if ph_out_low is not None:
        ph_out_low = mag * np.exp(1j * np.angle(ph_out_low))
    return np.asarray(ph_out, dtype=np.complex64), ph_out_low
# ---------------------------------------------------------------------------
# uw_grid_wrapped — resample to grid, optional filter (MATLAB uw_grid_wrapped.m)
# ---------------------------------------------------------------------------

@dataclass
class UwGrid:
    ph: np.ndarray
    ph_in: np.ndarray
    ph_lowpass: Optional[np.ndarray]
    ph_uw_predef: Optional[np.ndarray]
    xy: np.ndarray
    ij: np.ndarray
    nzix: np.ndarray
    grid_ij: np.ndarray
    grid_x_min: float
    grid_y_min: float
    n_i: int
    n_j: int
    n_ifg: int
    n_ps: int
    n_ps_orig: int
    pix_size: float


def uw_grid_wrapped(
    ph_in: np.ndarray,
    xy_in: np.ndarray,
    pix_size: float = 200,
    prefilt_win: int = 32,
    goldfilt_flag: str = "n",
    lowfilt_flag: str = "n",
    gold_alpha: float = 0.8,
    ph_in_predef: Optional[np.ndarray] = None,
) -> UwGrid:
    """Resample wrapped phase to grid; optional Goldstein/lowpass. Returns UwGrid."""
    n_ps, n_ifg = ph_in.shape
    logger.info("Resampling phase to grid...")
    logger.info("   Number of interferograms  : %d", n_ifg)
    logger.info("   Number of points per ifg  : %d", n_ps)
    xy_in = np.asarray(xy_in, dtype=np.float64)
    if xy_in.ndim == 1:
        xy_in = xy_in.reshape(-1, 2)
    if xy_in.shape[1] == 2:
        xy_in = np.column_stack([np.arange(1, n_ps + 1, dtype=np.float64), xy_in[:, 0], xy_in[:, 1]])
    predef_flag = ph_in_predef is not None and ph_in_predef.size > 0
    if pix_size == 0:
        grid_x_min = 1.0
        grid_y_min = 1.0
        n_i = int(xy_in[:, 2].max())
        n_j = int(xy_in[:, 1].max())
        grid_ij = np.column_stack([
            np.ceil((xy_in[:, 2] - grid_y_min + 1e-3)).astype(np.int32),
            np.ceil((xy_in[:, 1] - grid_x_min + 1e-3)).astype(np.int32),
        ])
    else:
        grid_x_min = float(xy_in[:, 1].min())
        grid_y_min = float(xy_in[:, 2].min())
        # MATLAB builds these as 1-based indices, folds the last boundary
        # cell into the preceding cell, then allocates exactly max(index)
        # rows/columns. Convert to 0-based only after that operation.
        grid_ij_1 = np.column_stack([
            np.ceil((xy_in[:, 2] - grid_y_min + 1e-3) / pix_size).astype(np.int32),
            np.ceil((xy_in[:, 1] - grid_x_min + 1e-3) / pix_size).astype(np.int32),
        ])
        for axis in (0, 1):
            max_ix = int(grid_ij_1[:, axis].max())
            if max_ix > 1:
                grid_ij_1[grid_ij_1[:, axis] == max_ix, axis] = max_ix - 1
        n_i = max(1, int(grid_ij_1[:, 0].max()))
        n_j = max(1, int(grid_ij_1[:, 1].max()))
        grid_ij = grid_ij_1 - 1
    prefilt_win = min(prefilt_win, n_i, n_j)
    if prefilt_win < 2:
        prefilt_win = 2
    if (goldfilt_flag == "y" or lowfilt_flag == "y") and min(n_i, n_j) < 4:
        goldfilt_flag = "n"
        lowfilt_flag = "n"
    nzix = None
    ph = None
    ph_lowpass = None
    ph_uw_predef = None
    for i1 in range(n_ifg):
        if np.isrealobj(ph_in):
            ph_this = np.exp(1j * np.asarray(ph_in[:, i1], dtype=np.float64))
        else:
            ph_this = np.asarray(ph_in[:, i1], dtype=np.complex64)
        ph_grid = np.zeros((n_i, n_j), dtype=np.complex128)
        if pix_size == 0:
            lin = (xy_in[:, 1].astype(np.int32) - 1) * n_i + xy_in[:, 2].astype(np.int32) - 1
            ph_grid.flat[lin] = ph_this
        else:
            for i in range(n_ps):
                gi, gj = grid_ij[i, 0], grid_ij[i, 1]
                ph_grid[gi, gj] += ph_this[i]
        if i1 == 0:
            nzix = ph_grid != 0
            n_ps_grid = int(nzix.sum())
            ph = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64)
            ph_lowpass = np.zeros((n_ps_grid, n_ifg), dtype=np.complex64) if lowfilt_flag == "y" else None
            ph_uw_predef = np.zeros((n_ps_grid, n_ifg), dtype=np.float32) if predef_flag else None
        if goldfilt_flag == "y" or lowfilt_flag == "y":
            ph_this_gold, ph_this_low = wrap_filt(
                np.asarray(ph_grid, dtype=np.complex64), prefilt_win, gold_alpha, lowfilt_flag
            )
            if lowfilt_flag == "y" and ph_lowpass is not None:
                ph_lowpass[:, i1] = ph_this_low.ravel(order="F")[nzix.ravel(order="F")]
            if goldfilt_flag == "y":
                ph[:, i1] = ph_this_gold.ravel(order="F")[nzix.ravel(order="F")]
            else:
                ph[:, i1] = ph_grid.ravel(order="F")[nzix.ravel(order="F")]
        else:
            ph[:, i1] = ph_grid.ravel(order="F")[nzix.ravel(order="F")]
    n_ps_orig = ph_in.shape[0]
    n_ps_grid = int(nzix.sum())
    logger.info("   Number of resampled points: %d", n_ps_grid)
    nz_flat = np.flatnonzero(nzix.ravel(order="F"))
    nz_i, nz_j = np.unravel_index(nz_flat, nzix.shape, order="F")
    if pix_size == 0:
        xy = xy_in
    else:
        xy = np.column_stack([
            np.arange(1, n_ps_grid + 1, dtype=np.float64),
            (nz_j + 0.5) * pix_size,
            (nz_i + 0.5) * pix_size,
        ])
    ij = np.column_stack([nz_i + 1, nz_j + 1])
    return UwGrid(
        ph=ph,
        ph_in=ph_in,
        ph_lowpass=ph_lowpass,
        ph_uw_predef=ph_uw_predef,
        xy=xy,
        ij=ij,
        nzix=nzix,
        grid_ij=grid_ij,
        grid_x_min=grid_x_min,
        grid_y_min=grid_y_min,
        n_i=n_i,
        n_j=n_j,
        n_ifg=n_ifg,
        n_ps=n_ps_grid,
        n_ps_orig=n_ps_orig,
        pix_size=pix_size,
    )


# ---------------------------------------------------------------------------
# uw_interp — Delaunay, edges, rowix/colix, Z (MATLAB uw_interp.m)
# ---------------------------------------------------------------------------

@dataclass
class UwInterp:
    edgs: np.ndarray
    n_edge: int
    rowix: np.ndarray
    colix: np.ndarray
    Z: np.ndarray


def uw_interp(uw: UwGrid) -> UwInterp:
    """Build triangulation and grid→node mapping. Uses Delaunay (no triangle binary)."""
    logger.info("Interpolating grid...")
    nrow, ncol = uw.nzix.shape
    node_flat = np.flatnonzero(uw.nzix.ravel(order="F"))
    y, x = np.unravel_index(node_flat, uw.nzix.shape, order="F")
    xy = np.column_stack([np.arange(1, uw.n_ps + 1), x + 1, y + 1]).astype(np.float64)
    tri = Delaunay(xy[:, 1:3])
    simplices = tri.simplices
    edges_set = set()
    for s in simplices:
        for i in range(3):
            a, b = s[i], s[(i + 1) % 3]
            edges_set.add((min(a, b), max(a, b)))
    edgs = np.array(sorted(edges_set), dtype=np.int32)
    n_edge = len(edgs)
    edgs = np.column_stack([np.arange(1, n_edge + 1), edgs[:, 0] + 1, edgs[:, 1] + 1])
    x_pts = xy[:, 1]
    y_pts = xy[:, 2]
    tree = cKDTree(np.column_stack([x_pts, y_pts]))
    X = np.arange(1, ncol + 1)
    Y = np.arange(1, nrow + 1)
    XX, YY = np.meshgrid(X, Y)
    query = np.column_stack([XX.ravel(), YY.ravel()])
    _, Z_flat = tree.query(query, k=1)
    Z = (Z_flat + 1).reshape(nrow, ncol).astype(np.int32)
    # MATLAB's Z(:) is column-major. Build horizontal (col) and vertical
    # (row) edges explicitly so this does not depend on NumPy flatten order.
    grid_edges_col = np.column_stack([
        Z[:, :-1].ravel(order="F"),
        Z[:, 1:].ravel(order="F"),
    ])
    grid_edges_row = np.column_stack([
        Z[:-1, :].ravel(order="C"),
        Z[1:, :].ravel(order="C"),
    ])
    grid_edges = np.vstack([grid_edges_col, grid_edges_row])
    sorted_edges = np.sort(grid_edges, axis=1)
    edge_sign = np.where(
        grid_edges[:, 0] == grid_edges[:, 1],
        0,
        np.where(grid_edges[:, 0] < grid_edges[:, 1], 1, -1),
    )
    normalized = sorted_edges.copy()
    normalized[normalized[:, 0] == normalized[:, 1], :] = 0
    unique_edges, inverse = np.unique(normalized, axis=0, return_inverse=True)
    nonzero = np.any(unique_edges != 0, axis=1)
    edge_number = np.zeros(unique_edges.shape[0], dtype=np.int32)
    edge_number[nonzero] = np.arange(1, np.count_nonzero(nonzero) + 1)
    edgs_nonzero = unique_edges[nonzero]
    edgs_final = np.column_stack([
        np.arange(1, edgs_nonzero.shape[0] + 1, dtype=np.int32),
        edgs_nonzero,
    ])
    grid_edge_ix = edge_number[inverse] * edge_sign
    n_col_edges = nrow * (ncol - 1)
    colix = grid_edge_ix[:n_col_edges].reshape(nrow, ncol - 1, order="F")
    rowix = grid_edge_ix[n_col_edges:].reshape(nrow - 1, ncol, order="C")
    n_edge = len(edgs_final)
    logger.info("   Number of unique edges in grid: %d", n_edge)
    return UwInterp(
        edgs=edgs_final,
        n_edge=n_edge,
        rowix=rowix.astype(np.float64),
        colix=colix.astype(np.float64),
        Z=Z,
    )


# ---------------------------------------------------------------------------
# uw_sb_unwrap_space_time — time smoothing, K/Kt, dph_space_uw, dph_noise (simplified 3D_QUICK)
# ---------------------------------------------------------------------------

@dataclass
class UwSpaceTime:
    dph_space_uw: np.ndarray
    dph_noise: np.ndarray
    spread: np.ndarray
    G: np.ndarray
    predef_ix: Optional[np.ndarray] = None


def _build_G(n_ifg: int, n_image: int, ifgday_ix: np.ndarray) -> np.ndarray:
    G = np.zeros((n_ifg, n_image), dtype=np.float64)
    for i in range(n_ifg):
        m, s = int(ifgday_ix[i, 0]), int(ifgday_ix[i, 1])
        if 0 <= m < n_image:
            G[i, m] = -1
        if 0 <= s < n_image:
            G[i, s] = 1
    return G


def uw_sb_unwrap_space_time(
    uw: UwGrid,
    ui: UwInterp,
    day: np.ndarray,
    ifgday_ix: np.ndarray,
    bperp: np.ndarray,
    unwrap_method: str = "3D_QUICK",
    time_win: float = 730,
    la_flag: str = "y",
    n_trial_wraps: float = 6,
    temp: Optional[np.ndarray] = None,
) -> UwSpaceTime:
    """Time-space unwrap: arc phase diff, optional K/Kt, time smooth, dph_noise. Simplified 3D path."""
    logger.info("Unwrapping in time-space...")
    n_ifg = uw.n_ifg
    edgs = ui.edgs
    if edgs.shape[1] >= 3:
        node_a = (edgs[:, 1].astype(np.int32) - 1).clip(0, uw.n_ps - 1)
        node_b = (edgs[:, 2].astype(np.int32) - 1).clip(0, uw.n_ps - 1)
    else:
        node_a = (edgs[:, 0].astype(np.int32) - 1).clip(0, uw.n_ps - 1)
        node_b = (edgs[:, 1].astype(np.int32) - 1).clip(0, uw.n_ps - 1)
    dph_space = uw.ph[node_b, :] * np.conj(uw.ph[node_a, :])
    dph_space = dph_space / (np.abs(dph_space) + 1e-12)
    n_image = day.size
    G = _build_G(n_ifg, n_image, ifgday_ix)
    nzc_ix = np.sum(np.abs(G), axis=0) != 0
    col_map = np.full(nzc_ix.size, -1, dtype=np.int32)
    col_map[nzc_ix] = np.arange(np.count_nonzero(nzc_ix), dtype=np.int32)
    ifgday_ix = col_map[np.asarray(ifgday_ix, dtype=np.int32)]
    day = day[nzc_ix]
    G = G[:, nzc_ix]
    n = G.shape[1]
    K = np.zeros(ui.n_edge, dtype=np.float32)
    Kt = np.zeros(ui.n_edge, dtype=np.float32)
    if la_flag == "y" and bperp.size >= 2:
        bperp_range = float(np.max(bperp) - np.min(bperp))
        dph_sub = dph_space
        bperp_sub = np.asarray(bperp, dtype=np.float64)
        trial_wraps = float(n_trial_wraps)

        sequential = np.flatnonzero(np.abs(np.diff(ifgday_ix, axis=1).ravel()) == 1)
        if sequential.size >= day.size - 1:
            dph_sub = dph_space[:, sequential]
            bperp_sub = bperp_sub[sequential]
        else:
            ifgs_per_image = np.sum(np.abs(G), axis=0)
            max_image = int(np.argmax(ifgs_per_image))
            if int(ifgs_per_image[max_image]) >= day.size - 2:
                use_ifg = G[:, max_image] != 0
                gsub = G[use_ifg, max_image]
                sign_ix = -np.sign(gsub)
                dph_chain = dph_space[:, use_ifg].copy()
                dph_chain[:, sign_ix == -1] = np.conj(dph_chain[:, sign_ix == -1])
                bp_chain = np.asarray(bperp[use_ifg], dtype=np.float64)
                bp_chain[sign_ix == -1] *= -1
                other_image = np.sum(ifgday_ix[use_ifg, :], axis=1) - max_image
                chain_images = np.concatenate([other_image, [max_image]])
                sort_ix = np.argsort(day[chain_images])
                dph_chain = np.column_stack([
                    dph_chain,
                    np.mean(np.abs(dph_chain), axis=1),
                ])[:, sort_ix]
                bp_chain = np.concatenate([bp_chain, [0.0]])[sort_ix]
                dph_sub = dph_chain[:, 1:] * np.conj(dph_chain[:, :-1])
                dph_sub /= np.abs(dph_sub) + 1e-12
                bperp_sub = np.diff(bp_chain)

        bperp_range_sub = float(np.max(bperp_sub) - np.min(bperp_sub))
        if bperp_range > 0 and bperp_range_sub > 0:
            trial_wraps *= bperp_range_sub / bperp_range
            if not np.isfinite(trial_wraps):
                trial_wraps = 6.0
            trial_mult = np.arange(
                -int(np.ceil(8 * trial_wraps)),
                int(np.ceil(8 * trial_wraps)) + 1,
                dtype=np.float64,
            )
            trial_phase = bperp_sub / bperp_range_sub * (np.pi / 4)
            trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
            coh = np.zeros(ui.n_edge, dtype=np.float32)

            for edge_i in range(ui.n_edge):
                cpx = dph_sub[edge_i, :].ravel()
                coh_trial = np.abs(np.sum(trial_phase_mat * cpx[:, None], axis=0))
                coh_trial /= np.sum(np.abs(cpx)) + 1e-12
                peak_i = int(np.argmax(coh_trial))
                coh_max = float(coh_trial[peak_i])

                falling = np.flatnonzero(np.diff(coh_trial[: peak_i + 1]) < 0)
                peak_start = int(falling[-1] + 1) if falling.size else 0
                rising = np.flatnonzero(np.diff(coh_trial[peak_i:]) > 0)
                peak_end = int(rising[0] + peak_i) if rising.size else coh_trial.size - 1
                other_peaks = coh_trial.copy()
                other_peaks[peak_start : peak_end + 1] = 0

                if coh_max - float(np.max(other_peaks)) > 0.1:
                    K0 = np.pi / 4 / bperp_range_sub * trial_mult[peak_i]
                    residual = cpx * np.exp(-1j * K0 * bperp_sub)
                    offset = np.sum(residual)
                    residual_angle = np.angle(residual * np.conj(offset))
                    weight = np.abs(cpx)
                    design = (weight * bperp_sub).reshape(-1, 1)
                    target = (weight * residual_angle).reshape(-1, 1)
                    correction = np.linalg.lstsq(design, target, rcond=None)[0][0, 0]
                    K[edge_i] = K0 + correction
                    phase_residual = cpx * np.exp(-1j * K[edge_i] * bperp_sub)
                    coh[edge_i] = np.abs(np.sum(phase_residual)) / (
                        np.sum(np.abs(phase_residual)) + 1e-12
                    )
            K[coh < 0.31] = 0
            dph_space = dph_space * np.exp(-1j * K[:, None] * bperp)
    spread = np.zeros((ui.n_edge, n_ifg), dtype=np.float32)
    if unwrap_method == "2D":
        dph_space_uw = np.angle(dph_space)
        if la_flag == "y":
            dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp
        dph_noise = np.zeros_like(dph_space_uw) * np.nan
    elif unwrap_method.upper() == "3D_FULL":
        dph_smooth_ifg = np.full(dph_space.shape, np.nan, dtype=np.float32)
        for image_i in range(n):
            use_ifg = G[:, image_i] != 0
            if np.count_nonzero(use_ifg) < n - 2:
                continue

            gsub = G[use_ifg, image_i]
            dph_sub = dph_space[:, use_ifg].copy()
            sign_ix = -np.sign(gsub).astype(np.float32)
            dph_sub[:, sign_ix == -1] = np.conj(dph_sub[:, sign_ix == -1])
            other_image = np.sum(ifgday_ix[use_ifg, :], axis=1) - image_i
            sort_ix = np.argsort(day[other_image])
            other_sorted = other_image[sort_ix]
            ifg_sorted = np.flatnonzero(use_ifg)[sort_ix]
            sign_sorted = sign_ix[sort_ix]
            dph_sub = dph_sub[:, sort_ix]
            dph_sub_angle = np.angle(dph_sub)
            day_sub = day[other_sorted]
            n_sub = day_sub.size
            dph_smooth = np.empty((ui.n_edge, n_sub), dtype=np.complex64)

            for date_i in range(n_sub):
                time_diff = day_sub[date_i] - day_sub
                weight = np.exp(-(time_diff ** 2) / (2 * time_win ** 2))
                weight /= np.sum(weight)
                dph_mean = dph_sub @ weight
                mean_adj = np.angle(
                    np.exp(1j * (dph_sub_angle - np.angle(dph_mean)[:, None]))
                )
                # A vector third argument to MATLAB lscov is a direct
                # observation-weight vector. Nearby dates must carry more,
                # not less, influence in this local linear fit.
                ls_weight = weight
                sw = float(np.sum(ls_weight))
                sx = float(np.dot(ls_weight, time_diff))
                sxx = float(np.dot(ls_weight, time_diff * time_diff))
                denom = sw * sxx - sx * sx
                sy = mean_adj @ ls_weight
                if abs(denom) > 1e-12:
                    sxy = mean_adj @ (ls_weight * time_diff)
                    intercept = (sxx * sy - sx * sxy) / denom
                else:
                    intercept = sy / sw
                dph_smooth[:, date_i] = dph_mean * np.exp(1j * intercept)

            increments = np.angle(
                dph_smooth[:, 1:] * np.conj(dph_smooth[:, :-1])
            )
            dph_smooth_uw = np.cumsum(
                np.column_stack([np.angle(dph_smooth[:, 0]), increments]),
                axis=1,
            )
            positive = np.flatnonzero(other_sorted - image_i > 0)
            close_i = int(positive[0]) if positive.size else n_sub - 1
            close_ix = [close_i - 1, close_i] if close_i > 0 else [close_i]
            close_phase = np.mean(dph_smooth_uw[:, close_ix], axis=1)
            dph_smooth_uw -= (
                close_phase - np.angle(np.exp(1j * close_phase))
            )[:, None]
            candidate = dph_smooth_uw * sign_sorted[None, :]

            already = ~np.isnan(dph_smooth_ifg[0, ifg_sorted])
            keep = np.ones(n_sub, dtype=bool)
            if np.any(already):
                cols = ifg_sorted[already]
                noise_old = np.std(
                    np.angle(dph_space[:, cols] * np.exp(-1j * dph_smooth_ifg[:, cols])),
                    axis=0, ddof=1,
                )
                noise_new = np.std(
                    np.angle(dph_space[:, cols] * np.exp(-1j * candidate[:, already])),
                    axis=0, ddof=1,
                )
                keep[np.flatnonzero(already)[noise_old < noise_new]] = False
            dph_smooth_ifg[:, ifg_sorted[keep]] = candidate[:, keep]

        dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))
        dph_noise[np.std(dph_noise, axis=1, ddof=1) > 1.2, :] = np.nan
        dph_space_uw = dph_smooth_ifg + dph_noise
        if la_flag == "y":
            dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp
    else:
        x = (day - day[0]) * (n - 1) / (day[-1] - day[0] + 1e-12)
        lstsq_res = np.linalg.lstsq(G[:, 1:], np.angle(dph_space).T, rcond=None)[0]
        dph_space_series = np.vstack([
            np.zeros((1, ui.n_edge), dtype=np.float64),
            lstsq_res,
        ])
        dph_smooth_series = np.zeros((n, ui.n_edge), dtype=np.float32)
        for i1 in range(n):
            w = np.exp(-((day[i1] - day) ** 2) / (2 * time_win ** 2))
            w /= w.sum()
            dph_smooth_series[i1, :] = (dph_space_series * w[:, np.newaxis]).sum(axis=0)
        dph_smooth_ifg = (G @ dph_smooth_series).T
        dph_noise = np.angle(dph_space * np.exp(-1j * dph_smooth_ifg))
        if unwrap_method in ("3D_SMALL_DEF", "3D_QUICK"):
            bad = np.nanstd(dph_noise, axis=1) > 1.3
            dph_noise[bad, :] = np.nan
        dph_space_uw = dph_smooth_ifg + dph_noise
        if la_flag == "y":
            dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp
    return UwSpaceTime(
        dph_space_uw=dph_space_uw.astype(np.float32),
        dph_noise=dph_noise.astype(np.float32),
        spread=spread,
        G=G,
    )


# ---------------------------------------------------------------------------
# uw_stat_costs — build cost file, call snaphu, read output (MATLAB uw_stat_costs.m)
# ---------------------------------------------------------------------------

_snaphu_cygwin_warned: set = set()  # work_dir str -> already logged Cygwin hint once per patch


def _writecpx(path: Path, ifgw: np.ndarray) -> None:
    """Write complex grid in Snaphu COMPLEX_DATA format.

    Mirrors MATLAB writecpx.m:
        vname_flt = zeros(nrow, ncol*2);
        vname_flt(:,1:2:end) = real(vname);
        vname_flt(:,2:2:end) = imag(vname);
        fwrite(fid, vname_flt.', 'float');   % col-major of transpose = row-major
    Python equivalent: build (nrow, ncol*2) interleaved, then write C-order (row-major).
    """
    v = np.asarray(ifgw, dtype=np.complex64)
    nrow, ncol = v.shape
    arr = np.empty((nrow, ncol * 2), dtype=np.float32)
    arr[:, 0::2] = v.real
    arr[:, 1::2] = v.imag
    with open(path, "wb") as f:
        # Row-major of arr = MATLAB fwrite(vname_flt.', 'float')
        arr.ravel(order='C').tofile(f)


def uw_stat_costs(
    uw: UwGrid,
    ui: UwInterp,
    ut: UwSpaceTime,
    unwrap_method: str,
    work_dir: Path,
    snaphu_path: str,
    variance: Optional[np.ndarray] = None,
    ifg_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Snaphu per IFG; return ph_uw (n_ps_grid, n_ifg), msd (n_ifg)."""
    import tempfile
    costscale = 100
    nshortcycle = 200
    maxshort = 32000
    logger.info("Unwrapping in space...")
    rowix = np.asarray(ui.rowix, dtype=np.float64)
    colix = np.asarray(ui.colix, dtype=np.float64)
    # Grid dimensions from uw so cost matrices match the actual grid (nzix)
    nrow, ncol = uw.nzix.shape
    node_flat = np.flatnonzero(uw.nzix.ravel(order="F"))
    y, x = np.unravel_index(node_flat, uw.nzix.shape, order="F")
    Z = ui.Z
    grid_edges = np.concatenate([
        colix[np.abs(colix) > 0].ravel(),
        rowix[np.abs(rowix) > 0].ravel(),
    ])
    n_edges = np.bincount(
        np.abs(grid_edges).astype(np.int32).clip(1, ui.n_edge),
        minlength=ui.n_edge + 1,
    )[1: ui.n_edge + 1]
    if n_edges.size < ui.n_edge:
        n_edges = np.resize(n_edges, ui.n_edge)
    if unwrap_method == "2D":
        edge_len = np.sqrt(
            (x[ui.edgs[:, 2].astype(int) - 1] - x[ui.edgs[:, 1].astype(int) - 1]) ** 2
            + (y[ui.edgs[:, 2].astype(int) - 1] - y[ui.edgs[:, 1].astype(int) - 1]) ** 2
        )
        pix_size = uw.pix_size if uw.pix_size > 0 else 5
        sigsq_noise = (2 * np.pi) ** 2 * (1 - np.exp(-edge_len * pix_size * 3 / 20000))
        sigsq_noise = sigsq_noise / 10
        dph_smooth = ut.dph_space_uw
    else:
        # MATLAB std(dph_noise, 0, 2) uses sample standard deviation and
        # propagates NaN. A NaN marks the complete edge as having no usable
        # statistics; nanstd would incorrectly re-enable that edge.
        with np.errstate(invalid="ignore"):
            sigsq_noise = (np.std(ut.dph_noise, axis=1, ddof=1) / (2 * np.pi)) ** 2
        dph_smooth = ut.dph_space_uw - ut.dph_noise
    nostats = np.isnan(sigsq_noise)
    sigsq_noise = np.nan_to_num(sigsq_noise, nan=1.0)
    rowix = rowix.copy()
    colix = colix.copy()
    for i in np.where(nostats)[0]:
        rowix[np.abs(rowix) == i + 1] = np.nan
        colix[np.abs(colix) == i + 1] = np.nan
    sigsq = np.int16(
        np.round((sigsq_noise * nshortcycle ** 2) / costscale * n_edges).clip(1, 32767)
    )
    rowcost = np.zeros((nrow - 1, ncol * 4), dtype=np.int16)
    colcost = np.zeros((nrow, (ncol - 1) * 4), dtype=np.int16)
    # Cost array layout per arc (4 int16 values, matching MATLAB 1-based → Python 0-based):
    #   [:,0::4] = offset    (MATLAB (:,1:4:end))
    #   [:,1::4] = sigsq     (MATLAB (:,2:4:end))
    #   [:,2::4] = dzmax     (MATLAB (:,3:4:end))
    #   [:,3::4] = laycost   (MATLAB (:,4:4:end))
    rowcost[:, 2::4] = maxshort  # dzmax
    colcost[:, 2::4] = maxshort  # dzmax
    lay_row = np.int16(np.isfinite(rowix)) * (-1 - maxshort) + 1
    lay_col = np.int16(np.isfinite(colix)) * (-1 - maxshort) + 1
    ncr = rowcost[:, 3::4].shape[1]  # ncol for rowcost
    ncc = colcost[:, 3::4].shape[1]  # ncol-1 for colcost
    rowcost[:, 3::4] = lay_row[:, :ncr].astype(np.int16)  # laycost
    colcost[:, 3::4] = lay_col[:, :ncc].astype(np.int16)  # laycost
    ph_uw = np.zeros((uw.n_ps, uw.n_ifg), dtype=np.float32)
    msd = np.zeros(uw.n_ifg, dtype=np.float32)
    conf_path = work_dir / "snaphu.conf"
    with open(conf_path, "w") as f:
        # INFILE and LINELENGTH are passed as positional args on the command line
        # (v2 rejects config INFILE + positional infile as "multiple input files")
        f.write("OUTFILE  snaphu.out\n")
        f.write("COSTINFILE snaphu.costinfile\n")
        f.write("STATCOSTMODE  DEFO\n")
        f.write("INFILEFORMAT  COMPLEX_DATA\n")
        f.write("OUTFILEFORMAT FLOAT_DATA\n")
    process_indices = (
        np.arange(uw.n_ifg, dtype=np.int32)
        if ifg_indices is None
        else np.asarray(ifg_indices, dtype=np.int32).ravel()
    )
    for i1 in process_indices:
        logger.info("   Processing IFG %d of %d", i1 + 1, uw.n_ifg)
        spread_i = np.asarray(ut.spread[:, i1], dtype=np.float32)
        spread_int = np.int16(
            np.round((np.abs(spread_i) * nshortcycle ** 2) / 6 / costscale * n_edges).clip(0, 32767)
        )
        sigsqtot = (sigsq + spread_int).astype(np.int16).clip(1, 32767)
        offset_cycle = (
            np.angle(np.exp(1j * ut.dph_space_uw[:, i1])) - dph_smooth[:, i1]
        ) / (2 * np.pi)
        valid_row = np.isfinite(rowix) & (rowix != 0)
        valid_col = np.isfinite(colix) & (colix != 0)
        eix_row = np.zeros(rowix.shape, dtype=np.int32)
        eix_col = np.zeros(colix.shape, dtype=np.int32)
        eix_row[valid_row] = np.abs(rowix[valid_row]).astype(np.int32) - 1
        eix_col[valid_col] = np.abs(colix[valid_col]).astype(np.int32) - 1
        offgrid_row = np.round(
            offset_cycle[eix_row] * np.sign(np.nan_to_num(rowix, nan=1)) * nshortcycle
        ).astype(np.int16)
        offgrid_row[~np.isfinite(rowix) | (rowix == 0)] = 0
        offgrid_col = np.round(
            offset_cycle[eix_col] * np.sign(np.nan_to_num(colix, nan=1)) * nshortcycle
        ).astype(np.int16)
        offgrid_col[~np.isfinite(colix) | (colix == 0)] = 0
        n_assign_r = min(ncr, offgrid_row.shape[1])
        n_assign_c = min(ncc, offgrid_col.shape[1])
        rowcost[:, 0::4][:, :n_assign_r] = -offgrid_row[:, :n_assign_r]  # offset
        colcost[:, 0::4][:, :n_assign_c] = offgrid_col[:, :n_assign_c]    # offset
        rowstdgrid = np.ones_like(rowix, dtype=np.int16)
        rowstdgrid[valid_row] = sigsqtot[eix_row[valid_row]]
        colstdgrid = np.ones_like(colix, dtype=np.int16)
        colstdgrid[valid_col] = sigsqtot[eix_col[valid_col]]
        rowcost[:, 1::4][:, :n_assign_r] = rowstdgrid[:, :n_assign_r]  # sigsq
        colcost[:, 1::4][:, :n_assign_c] = colstdgrid[:, :n_assign_c]  # sigsq
        cost_path = work_dir / "snaphu.costinfile"
        with open(cost_path, "wb") as f:
            # Row-major = MATLAB fwrite(rowcost','int16') (col-major of transpose)
            f.write(rowcost.astype(np.int16).tobytes(order='C'))
            f.write(colcost.astype(np.int16).tobytes(order='C'))
        # MATLAB uses reshape(uw.ph(Z,i1), nrow, ncol): every rectangular
        # grid cell is populated from its nearest PS grid node. Leaving empty
        # cells as zero removes residues and destroys the spatial unwrap.
        ifgw = uw.ph[np.asarray(Z, dtype=np.int32) - 1, i1].astype(
            np.complex64, copy=False
        )
        in_path = work_dir / "snaphu.in"
        _writecpx(in_path, ifgw)
        out_path = work_dir / "snaphu.out"
        log_path = work_dir / "snaphu.log"
        # Snaphu v1 & v2 compatible: positional infile + linelength, other params in config
        cmd = [snaphu_path, "-d", "-f", "snaphu.conf", "snaphu.in", str(ncol)]
        with open(log_path, "w") as logf:
            ret = subprocess.call(cmd, cwd=str(work_dir), stdout=logf, stderr=subprocess.STDOUT)
        if ret != 0:
            ifg_log = work_dir / f"snaphu.log.ifg_{i1 + 1}"
            log_text = ""
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                ifg_log.write_text(log_text, encoding="utf-8")
            except Exception:
                ifg_log = log_path
            logger.warning(
                "Snaphu returned %d for IFG %d. See %s",
                ret, i1 + 1, ifg_log,
            )
            try:
                if "cygwin" in log_text.lower() or "stackdump" in log_text.lower():
                    work_dir_key = str(work_dir)
                    if work_dir_key not in _snaphu_cygwin_warned:
                        _snaphu_cygwin_warned.add(work_dir_key)
                        logger.warning(
                            "Snaphu appears to be a Cygwin build (crashes on Windows with exit 34816). "
                            "Use a non-Cygwin Windows build, e.g. https://github.com/marsfan/SNAPHU-win , "
                            "and set SNAPHU_PATH (or snaphu_path) to its snaphu.exe."
                        )
            except Exception:
                pass
        # Read output if file exists with valid size, regardless of exit code.
        # MATLAB reads: fread(fid,[ncol,inf],'float') → transpose.
        # Python equivalent: row-major reshape (nrow, ncol).
        expected_bytes = nrow * ncol * 4
        if out_path.is_file() and out_path.stat().st_size >= expected_bytes:
            ifguw = np.fromfile(out_path, dtype=np.float32, count=nrow * ncol).reshape(nrow, ncol)
            diff1 = np.diff(ifguw, axis=0)
            diff2 = np.diff(ifguw, axis=1)
            nz1 = diff1[diff1 != 0]
            nz2 = diff2[diff2 != 0]
            msd[i1] = float(
                (np.sum(nz1 ** 2) + np.sum(nz2 ** 2))
                / (nz1.size + nz2.size + 1e-12)
            )
            ph_uw[:, i1] = ifguw[uw.nzix].ravel()
        elif ret != 0:
            logger.warning("   IFG %d: no valid snaphu.out produced", i1 + 1)
    return ph_uw, msd


# ---------------------------------------------------------------------------
# uw_unwrap_from_grid — map grid unwrapped back to original PS (MATLAB uw_unwrap_from_grid.m)
# ---------------------------------------------------------------------------

def uw_unwrap_from_grid(
    uw: UwGrid,
    ph_uw_grid: np.ndarray,
    msd_grid: np.ndarray,
    ph_in_orig: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map unwrapped phase from grid nodes back to original PS points."""
    logger.info("Unwrapping from grid...")
    n_ps_orig = getattr(uw, "n_ps_orig", ph_in_orig.shape[0])
    n_ifg = ph_in_orig.shape[1]
    gridix = np.zeros(uw.nzix.shape, dtype=np.int32)
    gridix[uw.nzix] = np.arange(uw.n_ps)
    ph_uw = np.zeros((n_ps_orig, n_ifg), dtype=np.float32)
    grid_ij = uw.grid_ij
    for i in range(n_ps_orig):
        if i >= grid_ij.shape[0]:
            ph_uw[i, :] = np.nan
            continue
        gi, gj = int(grid_ij[i, 0]), int(grid_ij[i, 1])
        if gi < 0 or gi >= gridix.shape[0] or gj < 0 or gj >= gridix.shape[1]:
            ph_uw[i, :] = np.nan
            continue
        ix = int(gridix[gi, gj])
        if ix < 0:
            ph_uw[i, :] = np.nan
        else:
            ph_uw_pix = ph_uw_grid[ix, :]
            if np.isrealobj(ph_in_orig):
                ph_uw[i, :] = ph_uw_pix + np.angle(
                    np.exp(1j * (ph_in_orig[i, :] - ph_uw_pix))
                )
            else:
                ph_uw[i, :] = ph_uw_pix + np.angle(
                    ph_in_orig[i, :] * np.exp(-1j * ph_uw_pix)
                )
    return ph_uw.astype(np.float32), msd_grid.astype(np.float32)
