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
    # Small Gaussian for smoothing magnitude
    from scipy.signal.windows import gaussian as gaussian_win
    B = np.outer(gaussian_win(7, 1.5), gaussian_win(7, 1.5))
    B /= B.sum()
    L = np.fft.ifftshift(
        np.outer(gaussian_win(n_win + n_pad, 2), gaussian_win(n_win + n_pad, 2))
    )
    L = L / (np.abs(L) + 1e-12)
    x = np.arange(1, n_win // 2 + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    wind_func = X + Y
    wind_func = np.hstack([wind_func, np.fliplr(wind_func)])
    wind_func = np.vstack([wind_func, np.flipud(wind_func)])
    for ix1 in range(n_win_i):
        for ix2 in range(n_win_j):
            i1 = ix1 * n_inc
            j1 = ix2 * n_inc
            i2 = min(i1 + n_win, n_i)
            j2 = min(j1 + n_win, n_j)
            i1 = i2 - n_win
            j1 = j2 - n_win
            i1 = max(0, i1)
            j1 = max(0, j1)
            ph_bit = np.zeros((n_win + n_pad, n_win + n_pad), dtype=np.complex128)
            ph_bit[:n_win, :n_win] = ph[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            from scipy.ndimage import convolve
            H = np.fft.ifftshift(convolve(np.fft.fftshift(H), B, mode="constant", cval=0))
            meanH = np.median(H)
            if meanH != 0:
                H = H / meanH
            H = np.power(H + 1e-12, alpha)
            ph_filt = np.fft.ifft2(ph_fft * H).real
            wf2 = wind_func[: i2 - i1, : j2 - j1] if wind_func.shape[0] >= (i2 - i1) else 1.0
            if np.isscalar(wf2):
                wf2 = np.ones((i2 - i1, j2 - j1))
            ph_out[i1:i2, j1:j2] += ph_filt[: i2 - i1, : j2 - j1] * wf2
            if low_flag == "y" and ph_out_low is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * L).real
                ph_out_low[i1:i2, j1:j2] += ph_filt_low[: i2 - i1, : j2 - j1] * wf2
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
        grid_ij = np.column_stack([
            np.ceil((xy_in[:, 2] - grid_y_min + 1e-3) / pix_size).astype(np.int32),
            np.ceil((xy_in[:, 1] - grid_x_min + 1e-3) / pix_size).astype(np.int32),
        ])
        grid_ij[:, 0] = np.clip(grid_ij[:, 0], 0, grid_ij[:, 0].max() - 1)
        grid_ij[:, 1] = np.clip(grid_ij[:, 1], 0, grid_ij[:, 1].max() - 1)
        n_i = int(grid_ij[:, 0].max()) + 1
        n_j = int(grid_ij[:, 1].max()) + 1
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
                ph_lowpass[:, i1] = ph_this_low[nzix].ravel()
            if goldfilt_flag == "y":
                ph[:, i1] = ph_this_gold[nzix].ravel()
            else:
                ph[:, i1] = ph_grid[nzix].ravel()
        else:
            ph[:, i1] = ph_grid[nzix].ravel()
    n_ps_orig = ph_in.shape[0]
    n_ps_grid = int(nzix.sum())
    logger.info("   Number of resampled points: %d", n_ps_grid)
    nz_i, nz_j = np.where(nzix)
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
    y, x = np.where(uw.nzix)
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
    Zvec = Z.ravel()
    n_row_edges = (nrow - 1) * ncol
    grid_edges_row = np.column_stack([Zvec[: -nrow], Zvec[nrow:]])
    Zvec_col = Z.T.ravel()
    n_col_edges = nrow * (ncol - 1)
    grid_edges_col = np.column_stack([Zvec_col[: -ncol], Zvec_col[ncol:]])
    sort_row = np.sort(grid_edges_row, axis=1)
    sort_col = np.sort(grid_edges_col, axis=1)
    edge_sign_row = np.sign(grid_edges_row[:, 1] - grid_edges_row[:, 0])
    edge_sign_col = np.sign(grid_edges_col[:, 1] - grid_edges_col[:, 0])
    all_edges = np.vstack([sort_row, sort_col])
    edge_sign = np.concatenate([edge_sign_row, edge_sign_col])
    same_ix = all_edges[:, 0] == all_edges[:, 1]
    all_edges[same_ix] = 0
    unq, inv = np.unique(all_edges, axis=0, return_inverse=True)
    zero_row = np.all(unq == 0, axis=1)
    unq = unq[~zero_row]
    edge_id = np.arange(1, len(unq) + 1)
    edgs_final = np.column_stack([edge_id, unq[:, 0], unq[:, 1]])
    inv = inv.copy()
    for i in range(len(zero_row)):
        if zero_row[i]:
            inv[inv == i] = -1
    mask = inv >= 0
    new_inv = np.full(inv.shape, -1)
    idx = np.where(~zero_row)[0]
    for i, old_i in enumerate(idx):
        new_inv[inv == old_i] = i
    grid_edge_ix = (new_inv + 1) * np.where(mask, edge_sign, 0)
    rowix = grid_edge_ix[:n_row_edges].reshape(nrow - 1, ncol)
    colix = grid_edge_ix[n_row_edges : n_row_edges + n_col_edges].reshape(nrow, ncol - 1)
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
    day = day[nzc_ix]
    G = G[:, nzc_ix]
    n = G.shape[1]
    ifgday_ix = ifgday_ix.copy()
    K = np.zeros(ui.n_edge, dtype=np.float32)
    Kt = np.zeros(ui.n_edge, dtype=np.float32)
    if la_flag == "y" and bperp.size >= 2:
        bperp_range = float(np.max(bperp) - np.min(bperp))
        if bperp_range > 0:
            trial_mult = np.arange(
                -int(np.ceil(8 * n_trial_wraps)),
                int(np.ceil(8 * n_trial_wraps)) + 1,
                dtype=np.float64,
            )
            trial_phase = bperp / bperp_range * (np.pi / 4)
            trial_phase_mat = np.exp(-1j * np.outer(trial_phase, trial_mult))
            for i in range(ui.n_edge):
                cpx = dph_space[i, :].ravel()
                phaser = trial_phase_mat * cpx[:, np.newaxis]
                coh_trial = np.abs(phaser.sum(axis=0)) / (np.abs(cpx).sum() + 1e-12)
                imax = np.argmax(coh_trial)
                coh_max = coh_trial[imax]
                coh_trial[max(0, imax - 1) : min(len(coh_trial), imax + 2)] = 0
                if coh_max - np.max(coh_trial) > 0.1:
                    K0 = np.pi / 4 / bperp_range * trial_mult[imax]
                    res = np.angle(cpx * np.exp(-1j * K0 * bperp))
                    w = np.abs(cpx) + 1e-12
                    mopt = np.linalg.lstsq(
                        (w * bperp).reshape(-1, 1), (w * res).reshape(-1, 1), rcond=None
                    )[0].ravel()[0]
                    K[i] = K0 + mopt
            dph_space = dph_space * np.exp(-1j * K[:, np.newaxis] * bperp)
    spread = np.zeros((ui.n_edge, n_ifg), dtype=np.float32)
    if unwrap_method == "2D":
        dph_space_uw = np.angle(dph_space)
        if la_flag == "y":
            dph_space_uw = dph_space_uw + K[:, np.newaxis] * bperp
        dph_noise = np.zeros_like(dph_space_uw) * np.nan
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
    y, x = np.where(uw.nzix)
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
        with np.errstate(invalid="ignore"):
            sigsq_noise = (np.nanstd(ut.dph_noise, axis=1) / (2 * np.pi)) ** 2
        sigsq_noise = np.nan_to_num(sigsq_noise, nan=1.0)
        dph_smooth = ut.dph_space_uw - np.nan_to_num(ut.dph_noise, nan=0)
    nostats = np.isnan(sigsq_noise)
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
    lay_row = np.int16(np.isfinite(rowix) & (rowix != 0)) * (-1 - maxshort) + 1
    lay_col = np.int16(np.isfinite(colix) & (colix != 0)) * (-1 - maxshort) + 1
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
    for i1 in range(uw.n_ifg):
        logger.info("   Processing IFG %d of %d", i1 + 1, uw.n_ifg)
        spread_i = np.zeros(ui.n_edge, dtype=np.float32)
        spread_int = np.int16(
            np.round((np.abs(spread_i) * nshortcycle ** 2) / 6 / costscale * n_edges).clip(0, 32767)
        )
        sigsqtot = (sigsq + spread_int).astype(np.int16).clip(1, 32767)
        offset_cycle = (
            np.angle(np.exp(1j * ut.dph_space_uw[:, i1])) - dph_smooth[:, i1]
        ) / (2 * np.pi)
        eix_row = (np.abs(rowix).astype(np.int32) - 1).clip(0, ui.n_edge - 1)
        eix_col = (np.abs(colix).astype(np.int32) - 1).clip(0, ui.n_edge - 1)
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
        rowstdgrid[np.isfinite(rowix) & (rowix != 0)] = sigsqtot[eix_row[np.isfinite(rowix) & (rowix != 0)]]
        colstdgrid = np.ones_like(colix, dtype=np.int16)
        colstdgrid[np.isfinite(colix) & (colix != 0)] = sigsqtot[eix_col[np.isfinite(colix) & (colix != 0)]]
        rowcost[:, 1::4][:, :n_assign_r] = rowstdgrid[:, :n_assign_r]  # sigsq
        colcost[:, 1::4][:, :n_assign_c] = colstdgrid[:, :n_assign_c]  # sigsq
        cost_path = work_dir / "snaphu.costinfile"
        with open(cost_path, "wb") as f:
            # Row-major = MATLAB fwrite(rowcost','int16') (col-major of transpose)
            f.write(rowcost.astype(np.int16).tobytes(order='C'))
            f.write(colcost.astype(np.int16).tobytes(order='C'))
        ifgw = np.zeros((nrow, ncol), dtype=np.complex64)
        ifgw[uw.nzix] = uw.ph[:, i1]
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
