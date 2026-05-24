#!/usr/bin/env python3
"""
phase_unwrapping.py — StaMPS Step 6: Phase unwrap (Python port)
================================================================

Implements the entry and overall flow of MATLAB ``ps_unwrap`` and
``sb_invert_uw``. All input data is read from HDF5 (.h5) produced by
Steps 1–5; .mat files are not used for computation (only for reference).

MATLAB correspondence
--------------------
+---------------------------------------+------------------------------------------+
| MATLAB                                | Python                                    |
+=======================================+==========================================+
| ps_unwrap()                           | UnwrapPipeline(patch_dir, psver).run()   |
| load psver; psname=['ps',num2str(…)]  | _get_psver(); ps from ps{ver}.h5          |
| bperp_mat from bp or ps.bperp         | _load_bperp_mat() from bp*.h5 or derived  |
| ph_w from pm.ph_patch or rc.ph_rc     | _load_ph_w() from pm*.h5 / rc*.h5         |
| uw_3d(…) → ph_uw, msd                 | _run_uw_3d() [stub until full impl]      |
| stamps_save(phuwname, ph_uw, msd)     | Save phuw_*.h5                            |
| sb_invert_uw                          | _run_sb_invert_uw() [stub]                |
+---------------------------------------+------------------------------------------+

Refactored from: ps_unwrap.m, uw_3d.m, sb_invert_uw.m (Andy Hooper, 2006+)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Sibling imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
import sys  # noqa: E402
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig  # noqa: E402

logger = logging.getLogger("stamps")


# ============================================================================
# Data loading — HDF5 only (no .mat for computation)
# ============================================================================

def _load_h5_only(patch_dir: Path, basename: str, keys: List[str]) -> Dict[str, np.ndarray]:
    """Load datasets from HDF5 only. Raises FileNotFoundError if .h5 missing.

    Step 6 uses only .h5 files produced by previous steps.
    """
    h5_path = patch_dir / (basename + ".h5")
    if not h5_path.is_file():
        raise FileNotFoundError(
            f"Step 6 requires {h5_path.name} (data from previous steps)."
        )
    data: Dict[str, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as hf:
        for k in keys:
            if k in hf:
                data[k] = np.asarray(hf[k][:])
    return data


def _parse_drop_ifg_index(getparm_value: Any, n_ifg: int) -> np.ndarray:
    """Return 0-based indices of IFGs to drop (exclude from unwrap).

    MATLAB: drop_ifg_index from getparm; unwrap_ifg_index = setdiff([1:ps.n_ifg], drop_ifg_index).
    """
    if getparm_value is None:
        return np.array([], dtype=np.int64)
    val = getparm_value
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "[]", "none", "None"):
            return np.array([], dtype=np.int64)
        try:
            import ast
            val = ast.literal_eval(val)
        except Exception:
            return np.array([], dtype=np.int64)
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val).ravel()
    else:
        arr = np.atleast_1d(val)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    try:
        ix = np.round(np.asarray(arr, dtype=np.float64)).astype(np.int64)
    except (ValueError, TypeError):
        return np.array([], dtype=np.int64)
    ix = ix[(ix >= 1) & (ix <= n_ifg)]
    return (ix - 1).astype(np.int64)  # 0-based


# ============================================================================
# UnwrapPipeline — ps_unwrap entry and overall flow
# ============================================================================

class UnwrapPipeline:
    """Step 6: Unwrap phase (entry + data prep + unwrap + save).

    MATLAB: ``ps_unwrap.m``. All inputs from .h5; output written to phuw_*.h5.
    The actual 3D unwrap (uw_grid_wrapped → uw_interp → uw_sb_unwrap_space_time
    → uw_stat_costs → uw_unwrap_from_grid) is invoked via _run_uw_3d(); currently
    a stub that raises NotImplementedError until those submodules are implemented.
    """

    def __init__(self, patch_dir: Union[str, Path], psver: Optional[int] = None) -> None:
        self.patch_dir = Path(patch_dir).resolve()
        self._psver: Optional[int] = psver
        self._cfg = StampsConfig(work_dir=self.patch_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    def _get_psver(self) -> int:
        """PS version from psver.h5 or default 2 (after Step 4/5)."""
        if self._psver is not None:
            return self._psver
        psver_path = self.patch_dir / "psver.h5"
        if psver_path.is_file():
            with h5py.File(str(psver_path), "r") as hf:
                if "psver" in hf:
                    return int(np.asarray(hf["psver"]).ravel()[0])
        return 2

    def _load_ps(self, v: int) -> Dict[str, Any]:
        """Load PS metadata from ps{v}.h5 only. Build ifgday_ix if missing."""
        required = ["n_ps", "n_ifg", "xy", "bperp", "day", "master_day", "master_ix"]
        raw = _load_h5_only(self.patch_dir, f"ps{v}", required)
        n_ps = int(np.asarray(raw["n_ps"]).ravel()[0])
        n_ifg = int(np.asarray(raw["n_ifg"]).ravel()[0])
        master_ix_1 = int(np.asarray(raw["master_ix"]).ravel()[0])
        master_day = int(np.asarray(raw["master_day"]).ravel()[0])
        xy = np.asarray(raw["xy"], dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.ravel()
        bperp = np.asarray(raw["bperp"], dtype=np.float32).ravel()
        day = np.asarray(raw["day"], dtype=np.float64).ravel()
        n_image = day.size
        ifgday_ix = None
        h5_path = self.patch_dir / f"ps{v}.h5"
        if h5_path.is_file():
            with h5py.File(str(h5_path), "r") as hf:
                if "ifgday_ix" in hf:
                    ifgday_ix = np.asarray(hf["ifgday_ix"][:], dtype=np.int32)
        if ifgday_ix is None or ifgday_ix.size == 0:
            small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"
            if not small_baseline:
                # MATLAB ps_unwrap.m:
                #   ifgday_ix = [ones(ps.n_ifg,1)*ps.master_ix, [1:ps.n_ifg]'];
                #   unwrap_ifg_index = setdiff(unwrap_ifg_index, master_ix)
                ifgday_ix = np.column_stack([
                    np.full(n_ifg, master_ix_1, dtype=np.int32),
                    np.arange(1, n_ifg + 1, dtype=np.int32),
                ])
            else:
                ifgday_ix = np.column_stack([
                    np.arange(1, n_ifg + 1, dtype=np.int32),
                    np.arange(2, n_ifg + 2, dtype=np.int32),
                ])
        if ifgday_ix.ndim == 1:
            ifgday_ix = ifgday_ix.reshape(-1, 2)
        if ifgday_ix.shape[0] > n_ifg:
            ifgday_ix = ifgday_ix[:n_ifg]
        elif ifgday_ix.shape[0] < n_ifg:
            small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"
            if not small_baseline:
                missing = np.arange(ifgday_ix.shape[0] + 1, n_ifg + 1, dtype=np.int32)
                extra = np.column_stack([
                    np.full(missing.size, master_ix_1, dtype=np.int32),
                    missing,
                ])
            else:
                extra = np.column_stack([
                    np.arange(ifgday_ix.shape[0] + 1, n_ifg + 1, dtype=np.int32),
                    np.arange(ifgday_ix.shape[0] + 2, n_ifg + 2, dtype=np.int32),
                ])
            ifgday_ix = np.vstack([ifgday_ix, extra])
        return {
            "n_ps": n_ps,
            "n_ifg": n_ifg,
            "xy": xy,
            "bperp": bperp,
            "day": day,
            "master_day": master_day,
            "master_ix": master_ix_1,
            "ifgday_ix": ifgday_ix,
        }

    def _load_bperp_mat(self, ps: Dict[str, Any], v: int) -> np.ndarray:
        """Build bperp_mat (n_ps, n_ifg or n_ps, n_image). From bp{v}.h5 or from ps.bperp."""
        n_ps = ps["n_ps"]
        n_ifg = ps["n_ifg"]
        small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"
        bp_h5 = self.patch_dir / f"bp{v}.h5"
        if bp_h5.is_file():
            with h5py.File(str(bp_h5), "r") as hf:
                bperp_mat = np.asarray(hf["bperp_mat"][:], dtype=np.float32)
            if not small_baseline and bperp_mat.shape[1] == n_ifg - 1:
                master_ix_0 = ps["master_ix"] - 1
                bperp_mat = np.insert(bperp_mat, master_ix_0, 0.0, axis=1)
            return bperp_mat
        bperp = ps["bperp"]
        if not small_baseline:
            master_ix_0 = ps["master_ix"] - 1
            bperp_1 = np.concatenate([
                bperp[:master_ix_0],
                np.zeros(1, dtype=np.float32),
                bperp[master_ix_0:],
            ])
        else:
            bperp_1 = bperp
        return np.broadcast_to(
            bperp_1.astype(np.float32).reshape(1, -1),
            (n_ps, bperp_1.size),
        ).copy()

    def _load_ph_w(self, ps: Dict[str, Any], bperp_mat: np.ndarray, v: int) -> np.ndarray:
        """Wrapped phase for unwrap input: from pm.ph_patch or rc.ph_rc (+ optional K_ps).

        MATLAB (ps_unwrap.m):
          if unwrap_patch_phase=='y': ph_w = pm.ph_patch ./ abs(pm.ph_patch); [insert 1 at master]
          else: ph_w = rc.ph_rc; if pm.K_ps, ph_w .= ph_w .* exp(j*K_ps*bperp_mat)
        """
        unwrap_patch_phase = str(self._cfg.getparm("unwrap_patch_phase")).strip().lower() == "y"
        small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"
        n_ps, n_ifg = ps["n_ps"], ps["n_ifg"]
        master_ix_0 = ps["master_ix"] - 1

        if unwrap_patch_phase:
            pm = _load_h5_only(self.patch_dir, f"pm{v}", ["ph_patch"])
            ph_w = np.asarray(pm["ph_patch"], dtype=np.complex64)
            ph_w = ph_w / (np.abs(ph_w) + 1e-12)
            if not small_baseline:
                ph_w = np.insert(ph_w, master_ix_0, np.ones(n_ps, dtype=np.complex64), axis=1)
        else:
            rc = _load_h5_only(self.patch_dir, f"rc{v}", ["ph_rc"])
            ph_w = np.asarray(rc["ph_rc"], dtype=np.complex64)
            pm_path = self.patch_dir / f"pm{v}.h5"
            if pm_path.is_file():
                with h5py.File(str(pm_path), "r") as hf:
                    if "K_ps" in hf:
                        K_ps = np.asarray(hf["K_ps"][:], dtype=np.float32).ravel()
                        if K_ps.size == n_ps:
                            ph_w = ph_w * np.exp(1j * (K_ps[:, None] * bperp_mat))
            ph_w = ph_w.astype(np.complex64)

        # Normalize (MATLAB: ph_w(ix)=ph_w(ix)./abs(ph_w(ix)))
        ix = np.abs(ph_w) > 1e-12
        ph_w[ix] = ph_w[ix] / np.abs(ph_w[ix])
        return ph_w

    def run(self) -> None:
        """Execute Step 6: load data, build ph_w, unwrap, and save phuw.

        MATLAB: ps_unwrap() then sb_invert_uw if small_baseline_flag=='y'.
        """
        logger.info("Phase-unwrapping...")
        v = self._get_psver()
        small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"

        # ---- Filenames (MATLAB ps_unwrap lines 31-45) ----
        if small_baseline:
            phuwname = f"phuw_sb{v}"
        else:
            phuwname = f"phuw{v}"

        # ---- Load PS (from .h5 only) ----
        ps = self._load_ps(v)
        n_ps, n_ifg = ps["n_ps"], ps["n_ifg"]

        # ---- unwrap_ifg_index ----
        drop_ifg_index = self._cfg.getparm("drop_ifg_index")
        drop_ix = _parse_drop_ifg_index(drop_ifg_index, n_ifg)
        unwrap_ifg_index = np.setdiff1d(np.arange(n_ifg), drop_ix)

        # ---- bperp_mat ----
        bperp_mat = self._load_bperp_mat(ps, v)

        # ---- ph_w ----
        ph_w = self._load_ph_w(ps, bperp_mat, v)

        # Optional scla/ramp/aps subtraction skipped in minimal path (doc: from .h5 when implemented)
        scla_subtracted = False
        ramp_subtracted = False

        # ---- Options for uw_3d ----
        options = {
            "master_day": ps["master_day"],
            "time_win": self._cfg.getparm("unwrap_time_win"),
            "unwrap_method": self._cfg.getparm("unwrap_method"),
            "grid_size": self._cfg.getparm("unwrap_grid_size"),
            "prefilt_win": self._cfg.getparm("unwrap_gold_n_win"),
            "goldfilt_flag": self._cfg.getparm("unwrap_prefilter_flag"),
            "gold_alpha": self._cfg.getparm("unwrap_gold_alpha"),
            "la_flag": self._cfg.getparm("unwrap_la_error_flag"),
            "scf_flag": self._cfg.getparm("unwrap_spatial_cost_func_flag"),
        }
        # Defaults if not in parms
        if options["time_win"] is None:
            options["time_win"] = 730
        if options["unwrap_method"] is None:
            options["unwrap_method"] = "3D_FULL" if small_baseline else "3D"
        if options["grid_size"] is None:
            options["grid_size"] = 200
        if options["prefilt_win"] is None:
            options["prefilt_win"] = 32
        if options["goldfilt_flag"] is None:
            options["goldfilt_flag"] = "n"
        if options["gold_alpha"] is None:
            options["gold_alpha"] = 0.8
        if options["la_flag"] is None:
            options["la_flag"] = "y"
        if options["scf_flag"] is None:
            options["scf_flag"] = "n"

        # ---- Exclude master from unwrap in PS mode (MATLAB line 224) ----
        if not small_baseline:
            master_ix_1 = ps["master_ix"]
            unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, [master_ix_1 - 1])

        # ---- day / ifgday_ix / bperp for uw_3d (subset of ifgs) ----
        day = ps["day"] - float(ps["master_day"])
        ifgday_ix = ps["ifgday_ix"]
        bperp_1d = np.asarray(ps["bperp"], dtype=np.float32).ravel()
        if len(unwrap_ifg_index) > 0:
            ifgday_ix_sub = ifgday_ix[unwrap_ifg_index]
            bperp_sub = bperp_1d[unwrap_ifg_index] if unwrap_ifg_index.max() < bperp_1d.size else bperp_1d
        else:
            ifgday_ix_sub = np.zeros((0, 2), dtype=np.int32)
            bperp_sub = np.zeros(0, dtype=np.float32)

        # ---- Call uw_3d (full chain: grid → interp → space_time → stat_costs → from_grid) ----
        ph_w_sub = ph_w[:, unwrap_ifg_index] if len(unwrap_ifg_index) > 0 else ph_w
        if ph_w_sub.size == 0:
            ph_uw_some = np.zeros((n_ps, 0), dtype=np.float32)
            msd_some = np.zeros(0, dtype=np.float32)
        else:
            ph_uw_some, msd_some = self._run_uw_3d(
                ph_w_sub,
                ps["xy"],
                day,
                ifgday_ix_sub,
                bperp_sub,
                options,
            )

        # ---- Assemble full ph_uw (n_ps, n_ifg), fill unwrapped columns ----
        ph_uw = np.zeros((n_ps, n_ifg), dtype=np.float32)
        msd = np.zeros(n_ifg, dtype=np.float32)
        if len(unwrap_ifg_index) > 0:
            ph_uw[:, unwrap_ifg_index] = ph_uw_some
            msd[unwrap_ifg_index] = msd_some

        # Stub: add back scla/ramp/aps if subtracted (would load from .h5)
        if scla_subtracted or ramp_subtracted:
            logger.debug("Add-back scla/ramp not yet implemented (would use .h5)")

        # unwrap_patch_phase: ph_uw += angle(rc*conj(ph_w)) (MATLAB 281-283)
        unwrap_patch_phase = str(self._cfg.getparm("unwrap_patch_phase")).strip().lower() == "y"
        if unwrap_patch_phase:
            rc = _load_h5_only(self.patch_dir, f"rc{v}", ["ph_rc"])
            ph_rc = np.asarray(rc["ph_rc"], dtype=np.complex64)
            pm = _load_h5_only(self.patch_dir, f"pm{v}", ["ph_patch"])
            ph_patch = np.asarray(pm["ph_patch"], dtype=np.complex64)
            small_baseline = str(self._cfg.getparm("small_baseline_flag")).strip().lower() == "y"
            master_ix_0 = ps["master_ix"] - 1
            if not small_baseline:
                ph_w_ref = np.insert(ph_patch, master_ix_0, np.ones(n_ps, dtype=np.complex64), axis=1)
            else:
                ph_w_ref = ph_patch / (np.abs(ph_patch) + 1e-12)
            ph_uw += np.angle(ph_rc * np.conj(ph_w_ref))

        # Zero out dropped IFGs (MATLAB 290)
        for j in drop_ix:
            ph_uw[:, j] = 0.0

        # ---- Save phuw to .h5 ----
        phuw_path = self.patch_dir / f"{phuwname}.h5"
        phuw_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(phuw_path), "w") as hf:
            hf.create_dataset("ph_uw", data=ph_uw.astype(np.float32), compression="gzip")
            hf.create_dataset("msd", data=msd.astype(np.float32))
        logger.info("Saved %s (ph_uw %s, msd %s)", phuw_path.name, str(ph_uw.shape), str(msd.shape))

        # ---- SB: invert to single-master time series ----
        if small_baseline:
            self._run_sb_invert_uw(ps, v, phuw_path, unwrap_ifg_index)

    def _run_uw_3d(
        self,
        ph: np.ndarray,
        xy: np.ndarray,
        day: np.ndarray,
        ifgday_ix: np.ndarray,
        bperp: np.ndarray,
        options: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run 3D unwrap: uw_grid_wrapped → uw_interp → uw_sb_unwrap_space_time → uw_stat_costs → uw_unwrap_from_grid.

        MATLAB: [ph_uw_some, msd_some] = uw_3d(ph_w(:,unwrap_ifg_index), ps.xy, day, ifgday_ix, ps.bperp, options).
        """
        from uw_core import (
            get_snaphu_path,
            uw_grid_wrapped,
            uw_interp,
            uw_sb_unwrap_space_time,
            uw_stat_costs,
            uw_unwrap_from_grid,
        )
        n_ps, n_ifg = ph.shape
        grid_size = int(options.get("grid_size", 200))
        prefilt_win = int(options.get("prefilt_win", 32))
        goldfilt_flag = str(options.get("goldfilt_flag", "n")).strip().lower()
        lowfilt_flag = "n"
        gold_alpha = float(options.get("gold_alpha", 0.8))
        time_win = float(options.get("time_win", 730))
        unwrap_method = str(options.get("unwrap_method", "3D_QUICK")).strip()
        la_flag = str(options.get("la_flag", "y")).strip().lower()
        n_trial_wraps = 6.0
        uw = uw_grid_wrapped(
            ph,
            xy,
            pix_size=grid_size,
            prefilt_win=prefilt_win,
            goldfilt_flag=goldfilt_flag,
            lowfilt_flag=lowfilt_flag,
            gold_alpha=gold_alpha,
            ph_in_predef=None,
        )
        ui = uw_interp(uw)
        day_1d = np.asarray(day).ravel()
        ifgday_ix_0 = np.asarray(ifgday_ix, dtype=np.int32)
        if ifgday_ix_0.ndim == 1:
            ifgday_ix_0 = np.column_stack([np.arange(n_ifg), np.arange(1, n_ifg + 1)])
        if np.any(ifgday_ix_0 >= 1):
            ifgday_ix_0 = (ifgday_ix_0 - 1).clip(0, day_1d.size - 1)
        bperp_1d = np.asarray(bperp, dtype=np.float32).ravel()
        ut = uw_sb_unwrap_space_time(
            uw,
            ui,
            day_1d,
            ifgday_ix_0,
            bperp_1d,
            unwrap_method=unwrap_method,
            time_win=time_win,
            la_flag=la_flag,
            n_trial_wraps=n_trial_wraps,
        )
        snaphu_path = get_snaphu_path(self._cfg)
        work_dir = self.patch_dir
        ph_uw_grid, msd_grid = uw_stat_costs(
            uw,
            ui,
            ut,
            unwrap_method,
            work_dir,
            snaphu_path,
            variance=None,
        )
        ph_uw_orig, msd_orig = uw_unwrap_from_grid(uw, ph_uw_grid, msd_grid, uw.ph_in)
        return ph_uw_orig, msd_orig

    def _run_sb_invert_uw(
        self,
        ps: Dict[str, Any],
        v: int,
        phuw_sb_path: Path,
        unwrap_ifg_index: np.ndarray,
    ) -> None:
        """Invert unwrapped SB ifg phase to single-master time series. Stub.

        MATLAB: sb_invert_uw. Would read phuw_sb.h5, ps*.h5, pm*.h5/rc*.h5; write phuw.h5, phuw_sb_res.h5.
        """
        logger.warning(
            "Step 6 sb_invert_uw not yet implemented; skipping inversion."
        )


# ============================================================================
# Module-level entry
# ============================================================================

def ps_unwrap(
    patch_dir: Union[str, Path],
    psver: Optional[int] = None,
) -> None:
    """Step 6 entry point (mirrors MATLAB ``ps_unwrap``).

    All data from .h5; output written to phuw_*.h5.
    """
    UnwrapPipeline(patch_dir=patch_dir, psver=psver).run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="StaMPS Step 6: Unwrap phase")
    parser.add_argument("patch_dir", type=str, help="Patch directory (contains ps2.h5, rc2.h5, etc.)")
    parser.add_argument("--psver", type=int, default=None, help="PS version (default: from psver.h5 or 2)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    ps_unwrap(Path(args.patch_dir), psver=args.psver)
