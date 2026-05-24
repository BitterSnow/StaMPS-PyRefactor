#!/usr/bin/env python3
"""
ps_correct_phase.py — Python port of ps_correct_phase.m  (StaMPS Step 5a)
==========================================================================

Correct the observed interferometric phase for the estimated look-angle
error (spatially-uncorrelated look-angle, SCLA).

MATLAB → Python mapping
-----------------------
+---------------------------------------------+------------------------------------------+
| MATLAB concept                              | Python mapping                           |
+=============================================+==========================================+
| ``ps_correct_phase``                        | ``PhaseCorrector(patch_dir).run()``      |
+---------------------------------------------+------------------------------------------+
| ``K_ps``  (DEM error ~ look-angle error)    | Same, loaded from ``pm2.h5`` / ``.mat`` |
+---------------------------------------------+------------------------------------------+
| ``bperp_mat`` (per-pixel perpendicular BL)  | Same, loaded from ``bp2.h5`` / ``.mat`` |
+---------------------------------------------+------------------------------------------+
| ``C_ps``  (master noise in PS mode)         | Same                                     |
+---------------------------------------------+------------------------------------------+
| ``ph_rc = ph .* exp(-j*(K_ps*bperp_mat))`` | NumPy vectorised multiplication          |
+---------------------------------------------+------------------------------------------+
| ``save('rc2', 'ph_rc' [, 'ph_reref'])``    | ``_save_rc_h5(…)``                       |
+---------------------------------------------+------------------------------------------+

Algorithm (ps_correct_phase.m)
------------------------------
**SB mode** (``small_baseline_flag == 'y'``)::

    ph_rc = ph .* exp(-j * (K_ps * bperp_mat))

**PS mode**::

    bperp_mat_full = [bperp_mat(:,1:master_ix-1), zeros, bperp_mat(:,master_ix:end)]
    ph_rc  = ph .* exp(-j * (K_ps * bperp_mat_full + C_ps))
    ph_reref = [ph_patch(:,1:master_ix-1), ones, ph_patch(:,master_ix:end)]

``ph_rc`` is the *range-corrected* phase — the raw phase with the estimated
topographic contribution removed.

Refactored from: ps_correct_phase.m (Andy Hooper, June 2006)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Union

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


# ============================================================================
# HDF5 persistence
# ============================================================================

def _save_rc_h5(
    h5_path: Path,
    *,
    ph_rc: np.ndarray,
    ph_reref: Optional[np.ndarray] = None,
) -> None:
    """Persist ``rc<ver>.h5``.

    MATLAB correspondence (ps_correct_phase.m lines 44/49)::

        save(rcname, 'ph_rc')                        % SB mode
        save(rcname, 'ph_rc', 'ph_reref')            % PS mode
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("ph_rc", data=ph_rc.astype(np.complex64),
                          compression="gzip")
        if ph_reref is not None:
            hf.create_dataset("ph_reref", data=ph_reref.astype(np.complex64),
                              compression="gzip")
    logger.info("Saved %s", h5_path.name)


# ============================================================================
# Data loading helpers
# ============================================================================

def _load_h5_or_mat(patch_dir: Path, basename: str, keys: list) -> dict:
    """Load datasets from HDF5 or MATLAB .mat file.

    Returns a dict mapping *key* → ndarray.
    """
    h5_path = patch_dir / (basename + ".h5")
    mat_path = patch_dir / (basename + ".mat")

    data: dict = {}
    if h5_path.is_file():
        with h5py.File(str(h5_path), "r") as hf:
            for k in keys:
                if k in hf:
                    data[k] = np.asarray(hf[k][:])
    elif mat_path.is_file():
        d = load_mat(mat_path)
        for k in keys:
            if k in d:
                data[k] = np.asarray(d[k])
    else:
        raise FileNotFoundError(
            f"Neither {h5_path.name} nor {mat_path.name} found in {patch_dir}"
        )
    return data


# ============================================================================
# Main class
# ============================================================================

class PhaseCorrector:
    """Correct interferometric phase for look-angle error.

    Parameters
    ----------
    patch_dir : Path
        Directory containing Step-4 outputs (version 2 files).
    psver : int
        PS version number (default 2 — after weeding).
    """

    def __init__(self, patch_dir: Union[str, Path], psver: int = 2) -> None:
        self.patch_dir = Path(patch_dir).resolve()
        self.psver = psver
        self._cfg = StampsConfig(work_dir=self.patch_dir)
        if not self._cfg._loaded:
            self._cfg.load()

    def run(self) -> None:
        """Execute the phase correction.

        MATLAB correspondence
        ---------------------
        ``ps_correct_phase.m`` (full function).
        """
        t0 = time.time()
        logger.info("Correcting phase for look angle error...")

        v = self.psver
        small_baseline_flag = str(
            self._cfg.getparm("small_baseline_flag")
        ).strip().lower()

        # ---- Load data ----
        ps = _load_h5_or_mat(
            self.patch_dir, f"ps{v}",
            ["n_ps", "n_ifg", "master_day", "day", "master_ix"],
        )
        n_ps = int(np.asarray(ps["n_ps"]).ravel()[0])
        n_ifg = int(np.asarray(ps["n_ifg"]).ravel()[0])
        # master_ix: MATLAB 1-based → keep as-is for column insertion logic
        master_ix_1 = int(np.asarray(ps["master_ix"]).ravel()[0])
        master_ix_0 = master_ix_1 - 1  # 0-based

        pm = _load_h5_or_mat(
            self.patch_dir, f"pm{v}",
            ["K_ps", "C_ps", "ph_patch"],
        )
        # K_ps, C_ps: ensure (n_ps,) 1-D
        K_ps = np.asarray(pm["K_ps"], dtype=np.float32).ravel()
        C_ps = np.asarray(pm["C_ps"], dtype=np.float32).ravel()
        ph_patch = np.asarray(pm.get("ph_patch", None))

        bp = _load_h5_or_mat(
            self.patch_dir, f"bp{v}",
            ["bperp_mat"],
        )
        bperp_mat = np.asarray(bp["bperp_mat"], dtype=np.float32)

        # ph — from ph<ver> or ps<ver>
        ph = self._load_ph(v)

        logger.info(
            "  n_ps=%d  n_ifg=%d  small_baseline=%s",
            n_ps, n_ifg, small_baseline_flag,
        )

        # ---- Phase correction (vectorised) ----
        if small_baseline_flag == "y":
            # MATLAB (line 43):
            #   ph_rc = ph .* exp(-j*(repmat(K_ps,1,n_ifg) .* bperp_mat))
            ph_rc = ph * np.exp(-1j * (K_ps[:, None] * bperp_mat))
            ph_reref = None
        else:
            # PS mode: insert zero column at master position in bperp_mat
            # MATLAB (line 46):
            #   bperp_mat = [bperp_mat(:,1:master_ix-1), zeros(n_ps,1,'single'),
            #                bperp_mat(:,master_ix:end)]
            bperp_mat_full = np.insert(
                bperp_mat, master_ix_0, 0.0, axis=1,
            ).astype(np.float32)

            # MATLAB (line 47):
            #   ph_rc = ph .* exp(-j*(K_ps*bperp_mat_full + C_ps))
            ph_rc = ph * np.exp(
                -1j * (K_ps[:, None] * bperp_mat_full + C_ps[:, None])
            )

            # MATLAB (line 48):
            #   ph_reref = [ph_patch(:,1:master_ix-1), ones(n_ps,1,'single'),
            #               ph_patch(:,master_ix:end)]
            ph_reref = np.insert(
                ph_patch.astype(np.complex64), master_ix_0,
                np.ones(n_ps, dtype=np.complex64), axis=1,
            )

        # ---- Save ----
        rc_path = self.patch_dir / f"rc{v}.h5"
        _save_rc_h5(rc_path, ph_rc=ph_rc, ph_reref=ph_reref)

        elapsed = time.time() - t0
        logger.info("Phase correction complete (%.1f s)", elapsed)

    # ------------------------------------------------------------------
    # ph loader
    # ------------------------------------------------------------------
    def _load_ph(self, ver: int) -> np.ndarray:
        """Load complex phase from ph<ver> or ps<ver>."""
        for src in [f"ph{ver}", f"ps{ver}"]:
            h5 = self.patch_dir / (src + ".h5")
            mat = self.patch_dir / (src + ".mat")
            if h5.is_file():
                with h5py.File(str(h5), "r") as hf:
                    if "ph" in hf:
                        return np.asarray(hf["ph"][:])
            if mat.is_file():
                d = load_mat(mat)
                if "ph" in d:
                    return np.asarray(d["ph"])
        raise FileNotFoundError(f"ph data not found for version {ver}")


# ============================================================================
# Module-level convenience
# ============================================================================

def ps_correct_phase(
    patch_dir: Union[str, Path],
    psver: int = 2,
) -> None:
    """Module-level entry point (mirrors MATLAB ``ps_correct_phase``)."""
    PhaseCorrector(patch_dir=patch_dir, psver=psver).run()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="StaMPS Step 5a: Correct phase for look-angle error",
    )
    parser.add_argument("patch_dir", type=str, help="Path to patch directory")
    parser.add_argument("--psver", type=int, default=2)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    ps_correct_phase(args.patch_dir, psver=args.psver)
