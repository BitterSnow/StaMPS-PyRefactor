#!/usr/bin/env python3
"""
stamps_main.py — StaMPS/MTI main processing runner (Python port)
================================================================

This is the top-level entry point, equivalent to MATLAB ``stamps.m``.
It orchestrates the numbered processing steps via the ``StampsRunner`` class.

Currently implemented
---------------------
* **Step 1** — Initial load of ISCE PS data (``ps_load_initial_isce``).
* **Step 2** — Estimate gamma / coherence (``ps_est_gamma_quick``).
* **Step 3** — Select PS pixels (``ps_select``).
* **Step 4** — Weed out adjacent / noisy PS (``ps_weed``).
* **Step 5** — Correct phase & merge patches (``ps_correct_phase`` + ``ps_merge_patches`` + ``ps_calc_ifg_std``).

Step definitions (100 % compatible with the original StaMPS)
------------------------------------------------------------
The step numbering follows ``stamps.m`` lines 5-13 and
``stamps_processing_stage.m``:

+------+--------------------------------------------------------------+
| Step | Description                                                  |
+======+==============================================================+
|  0   | Continue from the last completed stage                       |
+------+--------------------------------------------------------------+
|  1   | Initial load of data                                         |
+------+--------------------------------------------------------------+
|  2   | Estimate gamma (phase noise)                                 |
+------+--------------------------------------------------------------+
|  3   | Select PS pixels                                             |
+------+--------------------------------------------------------------+
|  4   | Weed out adjacent pixels                                     |
+------+--------------------------------------------------------------+
|  5   | Correct wrapped phase for spatially-uncorrelated look angle  |
|      | error and merge patches                                      |
+------+--------------------------------------------------------------+
|  6   | Unwrap phase                                                 |
+------+--------------------------------------------------------------+
|  7   | Calculate spatially-correlated look angle (DEM) error        |
+------+--------------------------------------------------------------+
|  8   | Filter spatially-correlated noise                            |
+------+--------------------------------------------------------------+

MATLAB → Python mapping
-----------------------
+---------------------------------------+------------------------------------------+
| MATLAB concept                        | Python mapping                           |
+=======================================+==========================================+
| ``stamps(start, end)``                | ``StampsRunner(…).run(start, end)``      |
+---------------------------------------+------------------------------------------+
| ``cd(patchdir); ps_load_initial_isce``| ``ISCEPSLoader(patch_dir).load()``       |
+---------------------------------------+------------------------------------------+
| ``stamps_save('ps1', …)``            | ``_save_ps1_h5(loader, path)``           |
+---------------------------------------+------------------------------------------+
| ``save('no_ps_info.mat', …)``        | ``no_ps_info.h5``                        |
+---------------------------------------+------------------------------------------+
| ``setappdata(0, …)``                 | data persisted in HDF5 + loader instance |
+---------------------------------------+------------------------------------------+
| ``logit(msg)``                        | ``logging.getLogger("stamps")``          |
+---------------------------------------+------------------------------------------+

CLI usage
---------
::

    python stamps_main.py --start 1 --end 1 --config path/to/test_data

Refactored from: stamps.m (Andy Hooper, June 2006)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Sibling module imports
# ---------------------------------------------------------------------------
# When running as a script, ensure the stamps_python directory is on sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from getparm import StampsConfig          # noqa: E402
from data_loader import ISCEPSLoader, ISCESBLoader  # noqa: E402
from gamma_est import GammaEstimator      # noqa: E402
from ps_selection_final import PSSelector  # noqa: E402
from ps_weeding import PSWeeder           # noqa: E402
from ps_correct_phase import PhaseCorrector  # noqa: E402
from ps_merge_patches import PatchMerger, calc_ifg_std  # noqa: E402
from phase_unwrapping import UnwrapPipeline  # noqa: E402
from scla_estimation import SclaEstimator  # noqa: E402
from scn_filt import ScnFiltPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("stamps")

_BANNER = (
    "########################################\n"
    "####### StaMPS/MTI Python v0.1.0 #######\n"
    "########################################"
)

# ---------------------------------------------------------------------------
# Step registry — matches stamps.m lines 5-13 and stamps_processing_stage.m
# ---------------------------------------------------------------------------
STEP_NAMES: Dict[int, str] = {
    0: "Continue from last completed stage",
    1: "Initial load of data",
    2: "Estimate gamma",
    3: "Select PS pixels",
    4: "Weed out adjacent pixels",
    5: "Correct wrapped phase & merge patches",
    6: "Unwrap phase",
    7: "Calc spatially-correlated look angle (DEM) error",
    8: "Filter spatially-correlated noise",
}


# ============================================================================
# HDF5 persistence helpers
# ============================================================================

def _save_ps1_h5(loader: ISCEPSLoader, h5_path: Path) -> None:
    """Persist Step-1 outputs to a single ``ps1.h5`` file.

    MATLAB correspondence
    ---------------------
    The original code calls ``stamps_save`` multiple times to produce
    ``ps1.mat``, ``ph1.mat``, ``da1.mat``, ``hgt1.mat``, ``bp1.mat``,
    and ``psver.mat``.  We consolidate everything into one HDF5 file
    with datasets at the root group ``/``.

    For large matrices (``ph``, ``bperp_mat``) we enable gzip compression
    to reduce disk usage.

    Dataset layout inside ``ps1.h5``
    ---------------------------------
    /ph              complex64   (n_ps, n_cols)   — complex phase stack
    /ij              int32       (n_ps, 3)        — [ID, azimuth, range]
    /lonlat          float64     (n_ps, 2)        — [longitude, latitude]
    /xy              float32     (n_ps, 3)        — [ID, local_x, local_y]
    /bperp           float64     (n_image,)       — perpendicular baselines
    /bperp_mat       float32     (n_ps, n_image-1)— per-PS baselines
    /day             int32       (n_image,)       — MATLAB datenum (days since 0000-01-01)
    /day_ix          int32       (n_slave,)       — slave sort permutation
    /master_day      int32       scalar           — MATLAB datenum (days since 0000-01-01)
    /ifgday          int32       (n_ifg, 2)       — interferogram date pairs [SB mode only]
    /ifgday_ix       int32       (n_ifg, 2)       — interferogram image indices [SB mode only]
    /master_ix       int32       scalar           — master index (0-based)
    /n_ifg           int32       scalar
    /n_image         int32       scalar
    /n_ps            int32       scalar
    /sort_ix         int32       (n_ps,)          — spatial sort permutation
    /ll0             float64     (2,)             — lon/lat origin
    /calconst        float64     (n_slave,)       — amplitude calibration
    /hgt             float32     (n_ps,)          — heights [optional]
    /D_A             float64     (n_ps,)          — dispersion [optional]
    /psver           int32       scalar
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing Step-1 HDF5 output: %s", h5_path)

    # Note: ISCEPSLoader stores dates as Python ordinals (days since 0001-01-01).
    # For compatibility with MATLAB .mat files, we convert to MATLAB datenum
    # (days since 0000-01-01) by adding 366 before saving to HDF5.

    with h5py.File(str(h5_path), "w") as hf:
        # --- Large matrices: use gzip compression ---
        # MATLAB: ph is complex single (complex64), often the biggest array.
        # h5py does not natively support complex dtypes, so we store the
        # real and imaginary parts as a compound type, or more portably,
        # we save as two float32 channels.  However, h5py >=2.2 *does*
        # support np.complex64 natively via an opaque dtype.  We test and
        # fall back to split storage if needed.
        if loader.ph is not None:
            _write_complex_dataset(hf, "ph", loader.ph, compression="gzip")

        if loader.bperp_mat is not None:
            hf.create_dataset(
                "bperp_mat", data=loader.bperp_mat, compression="gzip"
            )

        # --- Moderate arrays ---
        if loader.ij is not None:
            hf.create_dataset("ij", data=loader.ij)
        if loader.lonlat is not None:
            hf.create_dataset("lonlat", data=loader.lonlat)
        if loader.xy is not None:
            hf.create_dataset("xy", data=loader.xy)
        if loader.bperp is not None:
            hf.create_dataset("bperp", data=loader.bperp)
        if loader.day is not None:
            # Convert Python ordinal to MATLAB datenum for compatibility
            # MATLAB datenum: 1 = 0000-01-01, Python ordinal: 1 = 0001-01-01
            # Difference = 366 days (year 0000 was a leap year)
            day_matlab = loader.day.astype(np.int32) + 366
            hf.create_dataset("day", data=day_matlab)
        if loader.day_ix is not None:
            # MATLAB stores day_ix as 1-based image indices.
            hf.create_dataset("day_ix", data=loader.day_ix.astype(np.int32) + 1)
        # SB mode fields
        if hasattr(loader, "ifgday") and loader.ifgday is not None:
            # Convert Python ordinals to MATLAB datenum
            ifgday_matlab = loader.ifgday.astype(np.int32) + 366
            hf.create_dataset("ifgday", data=ifgday_matlab)
        if hasattr(loader, "ifgday_ix") and loader.ifgday_ix is not None:
            # Convert 0-based Python indices to 1-based MATLAB indices
            ifgday_ix_matlab = loader.ifgday_ix.astype(np.int32) + 1
            hf.create_dataset("ifgday_ix", data=ifgday_ix_matlab)
        if loader.sort_ix is not None:
            hf.create_dataset("sort_ix", data=loader.sort_ix)
        if loader.ll0 is not None:
            hf.create_dataset("ll0", data=loader.ll0)
        if loader.calconst is not None:
            hf.create_dataset("calconst", data=loader.calconst)
        if loader.hgt is not None:
            hf.create_dataset("hgt", data=loader.hgt)
        if getattr(loader, "inc", None) is not None:
            hf.create_dataset("inc", data=loader.inc)
        if getattr(loader, "la", None) is not None:
            hf.create_dataset("la", data=loader.la)
        if loader.D_A is not None:
            hf.create_dataset("D_A", data=loader.D_A)

        # --- Scalars ---
        # Convert Python ordinal to MATLAB datenum for compatibility
        master_day_matlab = np.int32(loader.master_day) + 366
        hf.create_dataset("master_day", data=master_day_matlab)
        # Convert 0-based Python index to 1-based MATLAB index
        master_ix_matlab = np.int32(loader.master_ix) + 1
        hf.create_dataset("master_ix", data=master_ix_matlab)
        hf.create_dataset("n_ifg", data=np.int32(loader.n_ifg))
        hf.create_dataset("n_image", data=np.int32(loader.n_image))
        hf.create_dataset("n_ps", data=np.int32(loader.n_ps))
        hf.create_dataset("psver", data=np.int32(loader.psver))

    file_size_mb = h5_path.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB)", h5_path.name, file_size_mb)


def _save_step1_auxiliary_angles(loader: ISCEPSLoader, patch_dir: Path) -> None:
    """Persist Step-1 incidence/look angle sidecar files when available."""
    if getattr(loader, "inc", None) is not None:
        with h5py.File(str(patch_dir / f"inc{loader.psver}.h5"), "w") as hf:
            hf.create_dataset("inc", data=loader.inc.astype(np.float32, copy=False))
    if getattr(loader, "la", None) is not None:
        with h5py.File(str(patch_dir / f"la{loader.psver}.h5"), "w") as hf:
            hf.create_dataset("la", data=loader.la.astype(np.float32, copy=False))


def _write_complex_dataset(
    hf: h5py.File,
    name: str,
    data: np.ndarray,
    compression: Optional[str] = None,
) -> None:
    """Write a complex numpy array to an HDF5 dataset.

    h5py supports np.complex64 / np.complex128 natively as a compound type
    ``{r: float32, i: float32}``.  If the version is too old, we fall back
    to storing ``ph_real`` and ``ph_imag`` as separate float32 datasets.
    """
    try:
        hf.create_dataset(name, data=data, compression=compression)
    except TypeError:
        # Fallback: split real / imag
        logger.debug(
            "h5py does not support complex dtype directly; "
            "splitting into %s_real / %s_imag",
            name,
            name,
        )
        hf.create_dataset(
            f"{name}_real",
            data=data.real.astype(np.float32),
            compression=compression,
        )
        hf.create_dataset(
            f"{name}_imag",
            data=data.imag.astype(np.float32),
            compression=compression,
        )


def _save_no_ps_info(h5_path: Path, step_flags: np.ndarray) -> None:
    """Persist the ``no_ps_info`` tracking array.

    MATLAB correspondence
    ---------------------
    ``save('no_ps_info.mat', 'stamps_step_no_ps')``
    A 5-element int vector tracking whether each step (1-5) found 0 PS.
    """
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset("stamps_step_no_ps", data=step_flags)


def _load_no_ps_info(h5_path: Path) -> np.ndarray:
    """Load existing ``no_ps_info`` or create a fresh zero vector."""
    if h5_path.is_file():
        with h5py.File(str(h5_path), "r") as hf:
            return np.array(hf["stamps_step_no_ps"])
    return np.zeros(5, dtype=np.int32)


# ============================================================================
# Patch discovery — mirrors stamps.m lines 119-149
# ============================================================================

def _discover_patches(
    project_dir: Path,
    patch_list_file: str = "patch.list",
) -> List[Path]:
    """Discover PATCH_* subdirectories to process.

    MATLAB correspondence (stamps.m lines 119-149)
    ------------------------------------------------
    1. If ``patch.list`` exists, read directory names from it.
    2. Otherwise, ``dir('PATCH_*')``.
    3. If no patches found, return ``[project_dir]`` (process cwd as a single patch).
    """
    patch_list = project_dir / patch_list_file
    patches: List[Path] = []

    if patch_list.is_file():
        for line in patch_list.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name:
                candidate = project_dir / name
                if candidate.is_dir():
                    patches.append(candidate)
                else:
                    logger.warning("Patch dir in %s not found: %s", patch_list_file, name)
    else:
        # Glob for PATCH_* directories (case-insensitive on Windows)
        patches = sorted(
            p for p in project_dir.iterdir()
            if p.is_dir() and p.name.upper().startswith("PATCH_")
        )

    if not patches:
        logger.info("No patch subdirectories found; treating project dir as single patch")
        patches = [project_dir]
    else:
        logger.info("Discovered %d patch(es)", len(patches))

    return patches


# ============================================================================
# Processing stage detection — mirrors stamps_processing_stage.m
# ============================================================================

def _detect_completed_stage(patch_dir: Path) -> int:
    """Detect the last completed processing stage for a patch directory.

    Returns the step number (0 = nothing done, 1-4 = stage files found).

    MATLAB correspondence (stamps_processing_stage.m lines 65-80)
    ---------------------------------------------------------------
    ```matlab
    if exist('weed1.mat','file')==2     → stage 4
    elseif exist('select1.mat','file')  → stage 3
    elseif exist('pm1.mat','file')      → stage 2
    elseif exist('ps1.mat','file')      → stage 1
    ```
    We check for both ``.mat`` and ``.h5`` variants.
    """
    checks = [
        (4, ["weed1.mat", "weed1.h5"]),
        (3, ["select1.mat", "select1.h5"]),
        (2, ["pm1.mat", "pm1.h5"]),
        (1, ["ps1.mat", "ps1.h5"]),
    ]
    for stage, filenames in checks:
        for fn in filenames:
            if (patch_dir / fn).is_file():
                return stage
    return 0


# ============================================================================
# StampsRunner
# ============================================================================

class StampsRunner:
    """Main orchestrator for StaMPS processing steps.

    MATLAB correspondence
    ---------------------
    This class replaces the monolithic ``stamps.m`` function.

    In MATLAB, ``stamps(start_step, end_step)`` iterates over patches and
    calls step-specific functions sequentially.  Here, ``run(start, end)``
    does the same, with each step dispatched to a dedicated ``run_step_N``
    method.

    Only **ISCE** processor + **PS** (non-SB) mode is implemented.
    SB branches (``small_baseline_flag == 'y'``) and other processors
    (gamma, doris, snap, gsar) are explicitly skipped.

    Parameters
    ----------
    project_dir : str or Path
        Top-level directory containing ``parms.mat``, ``patch.list`` and
        ``PATCH_*`` subdirectories (or the data files directly if running
        without patches).
    """

    def __init__(self, project_dir: Union[str, Path]) -> None:
        self.project_dir = Path(project_dir).resolve()

        # --- Configuration ---
        # Initialise (or reuse) the StampsConfig singleton
        StampsConfig.reset()
        self._config = StampsConfig(work_dir=self.project_dir)
        self._config.load()

        # Validate: only ISCE PS mode
        self._insar_processor: str = str(
            self._config.getparm("insar_processor") or "isce"
        ).strip().lower()
        self._small_baseline_flag: str = str(
            self._config.getparm("small_baseline_flag") or "n"
        ).strip().lower()

        logger.info("InSAR processor : %s", self._insar_processor)
        logger.info("Small baseline  : %s", self._small_baseline_flag)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def run(self, start_step: int = 1, end_step: int = 1) -> None:
        """Execute processing steps from *start_step* to *end_step*.

        MATLAB correspondence (stamps.m lines 77-83, 160-425)
        -------------------------------------------------------
        ```matlab
        for i = 1:length(patchdir)
            cd(patchdir(i).name)
            if start_step == 1 … end
            if start_step <= 2 & end_step >= 2 … end
            ...
        end
        ```
        """
        logger.info("\n%s", _BANNER)
        logger.info(
            "Running steps %d → %d  (project: %s)",
            start_step,
            end_step,
            self.project_dir,
        )

        if start_step < 0 or end_step > 8 or start_step > end_step:
            raise ValueError(
                f"Invalid step range [{start_step}, {end_step}]; "
                "must satisfy 0 <= start <= end <= 8"
            )

        # Discover patches (stamps.m lines 119-149)
        patches = _discover_patches(self.project_dir)
        per_patch_end = min(end_step, 5)

        if start_step <= per_patch_end:
            for patch_dir in patches:
                logger.info(
                    "\n======== Processing patch: %s ========", patch_dir.name
                )

                actual_start = start_step

                # Step 0: auto-detect start (stamps.m lines 179-205)
                if start_step == 0:
                    completed = _detect_completed_stage(patch_dir)
                    actual_start = completed + 1
                    if actual_start > per_patch_end:
                        logger.info(
                            "%s: already completed up to stage %d (per-patch end=%d); skipping",
                            patch_dir.name,
                            completed,
                            per_patch_end,
                        )
                        continue
                    logger.info(
                        "%s: resuming from stage %d (last completed: %d)",
                        patch_dir.name,
                        actual_start,
                        completed,
                    )

                # Dispatch per-patch steps only: Steps 1-4 and Step 5a.
                for step in range(actual_start, per_patch_end + 1):
                    step_name = STEP_NAMES.get(step, f"Step {step}")
                    logger.info(
                        "\n---- Step %d: %s [%s] ----",
                        step,
                        step_name,
                        patch_dir.name,
                    )
                    t0 = time.time()
                    self._dispatch_step(step, patch_dir)
                    elapsed = time.time() - t0
                    logger.info(
                        "Step %d completed in %.1f s", step, elapsed
                    )

        # --- Post-patch (project-level) processing ---
        # Step 5b: merge patches + IFG std (stamps.m lines 432-491)
        # This runs ONCE at the project level after all per-patch steps.
        if start_step <= 5 and end_step >= 5:
            t0 = time.time()
            self._run_step_5b()
            logger.info("Step 5b (merge + ifg_std) completed in %.1f s",
                        time.time() - t0)

        # --- Project-level post-merge processing ---
        # MATLAB runs ps_unwrap / ps_calc_scla / ps_scn_filt from the current
        # project directory after patch merge, not inside each PATCH_* folder.
        if end_step >= 6:
            project_start = 6 if start_step == 0 else max(start_step, 6)
            for step in range(project_start, end_step + 1):
                step_name = STEP_NAMES.get(step, f"Step {step}")
                logger.info(
                    "\n======== Step %d: %s [%s] ========",
                    step,
                    step_name,
                    self.project_dir.name,
                )
                t0 = time.time()
                self._dispatch_step(step, self.project_dir)
                logger.info("Step %d completed in %.1f s", step, time.time() - t0)

        logger.info("\nAll steps finished.")

    # ------------------------------------------------------------------
    # Step dispatch
    # ------------------------------------------------------------------
    def _dispatch_step(self, step: int, patch_dir: Path) -> None:
        """Route to the appropriate ``run_step_N`` method."""
        dispatch = {
            1: self.run_step_1,
            2: self.run_step_2,
            3: self.run_step_3,
            4: self.run_step_4,
            5: self.run_step_5,
            6: self.run_step_6,
            7: self.run_step_7,
            8: self.run_step_8,
        }
        handler = dispatch.get(step)
        if handler is None:
            logger.warning(
                "Step %d is not yet implemented; skipping.", step
            )
            return
        handler(patch_dir)

    # ------------------------------------------------------------------
    # Step 1: Initial load of data
    # ------------------------------------------------------------------
    def run_step_1(self, patch_dir: Path) -> None:
        """Step 1 — Load ISCE PS data and persist to HDF5.

        MATLAB correspondence (stamps.m lines 210-301)
        ------------------------------------------------
        The MATLAB code has nested branches:

        ```matlab
        if strcmpi(small_baseline_flag, 'y')
            if strcmpi(insar_processor, 'gamma') | strcmpi(…, 'snap')
                sb_load_initial_gamma;
            elseif strcmpi(…, 'isce')
                sb_load_initial_isce(data_inc);     ← SB + ISCE
            ...
        else
            if strcmpi(…, 'gamma') | strcmpi(…, 'snap')
                ps_load_initial_gamma;
            elseif strcmpi(…, 'isce')
                ps_load_initial_isce(data_inc);     ← PS + ISCE  ★ this path
            ...
        end
        ```

        We only implement the **PS + ISCE** path (marked ★ above).
        All other processors (gamma, doris, snap, gsar) and the
        SB branches are intentionally excluded.

        Sequence of operations
        ----------------------
        1. Validate that processor is ``isce``.
        2. Create ``ISCEPSLoader`` for the *patch_dir*.
        3. Call ``loader.load()`` — reads all binary/text files.
        4. Serialise results to ``ps1.h5`` via ``_save_ps1_h5``.
        5. Update ``no_ps_info.h5`` tracking flags (all zeros = success).

        The loaded data that MATLAB would store via ``setappdata(0, …)``
        is now written to the HDF5 file.  Downstream steps can reload from
        ``ps1.h5`` without re-parsing raw ISCE files.
        """
        # --- Processor gate (stamps.m lines 262-284) ---
        if self._insar_processor != "isce":
            raise NotImplementedError(
                f"Processor '{self._insar_processor}' is not supported. "
                "Only 'isce' is implemented in this Python port."
            )

        # --- no_ps_info initialisation (stamps.m lines 172-175) ---
        # MATLAB:
        #   if exist('no_ps_info.mat','file') ~= 2
        #       stamps_step_no_ps = zeros([5 1]);
        #       save('no_ps_info.mat', 'stamps_step_no_ps')
        #   end
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)

        existing_ps1 = patch_dir / "ps1.h5"
        if existing_ps1.is_file():
            with h5py.File(str(existing_ps1), "r") as hf:
                n_ps = int(np.asarray(hf["n_ps"]).ravel()[0]) if "n_ps" in hf else 0
            stamps_step_no_ps[:] = 0
            stamps_step_no_ps[0] = 0 if n_ps > 0 else 1
            _save_no_ps_info(no_ps_path, stamps_step_no_ps)
            logger.info(
                "Step 1 output already exists for %s (%d PS); reusing %s",
                patch_dir.name,
                n_ps,
                existing_ps1.name,
            )
            return

        # --- Core: call loader based on mode (stamps.m lines 218-301) ---
        # MATLAB:
        #   if strcmpi(small_baseline_flag, 'y')
        #       sb_load_initial_isce(data_inc)
        #   else
        #       ps_load_initial_isce(data_inc)
        #   end
        if self._small_baseline_flag == "y":
            logger.info("SB mode: Calling ISCESBLoader for: %s", patch_dir)
            loader = ISCESBLoader(work_dir=patch_dir)
            loader.load()
        else:
            logger.info("PS mode: Calling ISCEPSLoader for: %s", patch_dir)
            loader = ISCEPSLoader(work_dir=patch_dir)
            loader.load()

        # --- Persist to HDF5 ---
        # Output directory: same patch dir, file named ps1.h5
        h5_out = patch_dir / f"ps{loader.psver}.h5"
        _save_ps1_h5(loader, h5_out)
        _save_step1_auxiliary_angles(loader, patch_dir)

        # --- Update no_ps_info (stamps.m lines 243-244, 285-287) ---
        # MATLAB: stamps_step_no_ps(1:end) = 0;   (reset on re-processing)
        stamps_step_no_ps[:] = 0
        _save_no_ps_info(no_ps_path, stamps_step_no_ps)
        logger.info("Step 1 complete for %s", patch_dir.name)

    # ------------------------------------------------------------------
    # Step 2: Estimate gamma
    # ------------------------------------------------------------------
    def run_step_2(self, patch_dir: Path) -> None:
        """Step 2 — Estimate coherence (gamma) for PS candidate pixels.

        MATLAB correspondence (stamps.m lines 309-334)
        ------------------------------------------------
        ```matlab
        if start_step <= 2 & end_step >= 2
            load('no_ps_info.mat');
            stamps_step_no_ps(2:end) = 0;
            if stamps_step_no_ps(1) == 0
                if strcmpi(quick_est_gamma_flag, 'y')
                    ps_est_gamma_quick(est_gamma_parm);
                else
                    ps_est_gamma(est_gamma_parm);
                end
            else
                stamps_step_no_ps(2) = 1;
            end
            save('no_ps_info.mat', 'stamps_step_no_ps')
        end
        ```

        This Python port only implements ``ps_est_gamma_quick`` (the fast
        estimator).  The legacy ``ps_est_gamma`` is intentionally excluded.

        Sequence of operations
        ----------------------
        1. Load ``no_ps_info.h5`` to check if Step 1 found any PS.
        2. If Step 1 had PS (``stamps_step_no_ps[0] == 0``), run
           ``GammaEstimator`` which:
           a. Reads ``ps1.h5`` (phase, baselines, coordinates, …).
           b. Simulates random coherence distribution.
           c. Iterates: CLAP filter → topofit → convergence check.
           d. Writes ``pm1.h5`` with gamma estimates.
        3. Update ``no_ps_info.h5`` tracking flags.
        """
        # --- no_ps_info check (stamps.m lines 319-321) ---
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)
        # Reset flags for steps 2+ (re-processing)
        stamps_step_no_ps[1:] = 0

        if stamps_step_no_ps[0] != 0:
            logger.info(
                "No PS left from Step 1 for %s — skipping Step 2.",
                patch_dir.name,
            )
            stamps_step_no_ps[1] = 1
            _save_no_ps_info(no_ps_path, stamps_step_no_ps)
            return

        # --- Run gamma estimation (stamps.m lines 325-329) ---
        # MATLAB: quick_est_gamma_flag is always 'y' in this port
        logger.info("Running GammaEstimator (ps_est_gamma_quick) for %s", patch_dir.name)
        estimator = GammaEstimator(patch_dir=patch_dir, psver=1)
        estimator.run(restart_flag=0)

        # --- Update no_ps_info ---
        _save_no_ps_info(no_ps_path, stamps_step_no_ps)
        logger.info("Step 2 complete for %s", patch_dir.name)

    # ------------------------------------------------------------------
    # Step 3: Select PS pixels
    # ------------------------------------------------------------------
    def run_step_3(self, patch_dir: Path) -> None:
        """Step 3 — Select PS pixels based on gamma and D_A.

        MATLAB correspondence (stamps.m lines 337-365)
        ------------------------------------------------
        ```matlab
        if start_step <= 3 & end_step >= 3
            load('no_ps_info.mat');
            stamps_step_no_ps(3:end) = 0;
            if stamps_step_no_ps(2) == 0
                if strcmpi(quick_est_gamma_flag, 'y') &
                   strcmpi(reest_gamma_flag, 'y')
                    ps_select;           % ← re-estimation enabled
                else
                    ps_select(1);        % ← skip re-estimation
                end
            else
                stamps_step_no_ps(3) = 1;
            end
            save('no_ps_info.mat', 'stamps_step_no_ps')
        end
        ```

        This Python port dispatches to ``PSSelector`` with:
          * ``reest_flag=0`` when both ``quick_est_gamma_flag`` and
            ``select_reest_gamma_flag`` are ``'y'``.
          * ``reest_flag=1`` otherwise (skip re-estimation).
        """
        # --- no_ps_info check (stamps.m lines 349-351) ---
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)
        stamps_step_no_ps[2:] = 0

        if stamps_step_no_ps[1] != 0:
            logger.info(
                "No PS left from Step 2 for %s — skipping Step 3.",
                patch_dir.name,
            )
            stamps_step_no_ps[2] = 1
            _save_no_ps_info(no_ps_path, stamps_step_no_ps)
            return

        # --- Determine re-estimation flag (stamps.m lines 355-359) ---
        quick_flag = str(self._config.getparm("quick_est_gamma_flag") or "y").strip().lower()
        reest_flag_str = str(self._config.getparm("select_reest_gamma_flag") or "y").strip().lower()

        if quick_flag == "y" and reest_flag_str == "y":
            reest_flag = 0  # ps_select() — with re-estimation
        else:
            reest_flag = 1  # ps_select(1) — skip re-estimation

        logger.info(
            "Running PSSelector (ps_select) for %s (reest_flag=%d)",
            patch_dir.name, reest_flag,
        )
        selector = PSSelector(patch_dir=patch_dir, psver=1)
        selector.run(reest_flag=reest_flag)

        # --- Update no_ps_info ---
        _save_no_ps_info(no_ps_path, stamps_step_no_ps)
        logger.info("Step 3 complete for %s", patch_dir.name)

    # ------------------------------------------------------------------
    # Step 4: Weed out adjacent / noisy PS
    # ------------------------------------------------------------------
    def run_step_4(self, patch_dir: Path) -> None:
        """Step 4 — Weed out neighbouring and noisy PS pixels.

        MATLAB correspondence (stamps.m lines 367-395)
        ------------------------------------------------
        ```matlab
        if start_step <= 4 & end_step >= 4
            load('no_ps_info.mat');
            stamps_step_no_ps(4:end) = 0;
            if stamps_step_no_ps(3) == 0
                if strcmpi(small_baseline_flag, 'y')
                    ps_weed(0, 1);       % SB: skip adjacent weeding
                else
                    ps_weed;             % PS: full weeding
                end
            else
                stamps_step_no_ps(4) = 1;
            end
            save('no_ps_info.mat', 'stamps_step_no_ps')
        end
        ```

        For SB mode, ``ps_weed(0, 1)`` is called with ``no_weed_adjacent=1``
        (skip adjacent weeding, since SB interferograms don't have the same
        resolution-cell ambiguity as PS mode).
        """
        # --- no_ps_info check (stamps.m lines 377-379) ---
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)
        stamps_step_no_ps[3:] = 0

        if stamps_step_no_ps[2] != 0:
            logger.info(
                "No PS left from Step 3 for %s — skipping Step 4.",
                patch_dir.name,
            )
            stamps_step_no_ps[3] = 1
            _save_no_ps_info(no_ps_path, stamps_step_no_ps)
            return

        # --- Determine weeding mode (stamps.m lines 384-388) ---
        if self._small_baseline_flag == "y":
            logger.info("SB mode: ps_weed(0, 1) — skip adjacent weeding")
            no_weed_adjacent = 1
        else:
            logger.info("PS mode: ps_weed() — full weeding")
            no_weed_adjacent = None  # let config decide

        weeder = PSWeeder(patch_dir=patch_dir, psver=1)
        weeder.run(
            all_da_flag=0,
            no_weed_adjacent=no_weed_adjacent,
        )

        # --- Update no_ps_info ---
        _save_no_ps_info(no_ps_path, stamps_step_no_ps)
        logger.info("Step 4 complete for %s", patch_dir.name)

    # ------------------------------------------------------------------
    # Step 5a: Correct wrapped phase (per-patch)
    # ------------------------------------------------------------------
    def run_step_5(self, patch_dir: Path) -> None:
        """Step 5a — Correct phase for spatially-uncorrelated look-angle error.

        MATLAB correspondence (stamps.m lines 397-420)
        ------------------------------------------------
        ```matlab
        if start_step <= 5 & end_step >= 5
            load('no_ps_info.mat');
            stamps_step_no_ps(5:end) = 0;
            if stamps_step_no_ps(4) == 0
                ps_correct_phase;
            else
                stamps_step_no_ps(5) = 1;
            end
            save('no_ps_info.mat', 'stamps_step_no_ps')
        end
        ```

        This runs ``ps_correct_phase`` inside each patch directory,
        producing ``rc2.h5`` (range-corrected phase).

        The project-level merge (Step 5b) runs separately after all
        patches are processed — see ``_run_step_5b()``.
        """
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)
        stamps_step_no_ps[4:] = 0

        if stamps_step_no_ps[3] != 0:
            logger.info(
                "No PS left from Step 4 for %s — skipping Step 5a.",
                patch_dir.name,
            )
            stamps_step_no_ps[4] = 1
            _save_no_ps_info(no_ps_path, stamps_step_no_ps)
            return

        corrector = PhaseCorrector(patch_dir=patch_dir, psver=2)
        corrector.run()

        _save_no_ps_info(no_ps_path, stamps_step_no_ps)
        logger.info("Step 5a (phase correction) complete for %s", patch_dir.name)

    # ------------------------------------------------------------------
    # Step 6: Unwrap phase
    # ------------------------------------------------------------------
    def run_step_6(self, patch_dir: Path) -> None:
        """Step 6 — Unwrap phase (ps_unwrap; SB mode then sb_invert_uw).

        MATLAB correspondence (stamps.m lines 494-507)
        ------------------------------------------------
        ```matlab
        if start_step<=6 & end_step >=6
            ps_unwrap
            if strcmpi(small_baseline_flag,'y')
                sb_invert_uw
            end
        end
        ```

        All input data is read from .h5 (ps, rc, pm, bp); output is phuw_*.h5.
        The unwrap core (uw_3d) and sb_invert_uw are stubbed until full implementation.
        """
        # PS version from patch (psver.h5 or default 2 after Step 4/5)
        psver = 2
        psver_path = patch_dir / "psver.h5"
        if psver_path.is_file():
            with h5py.File(str(psver_path), "r") as hf:
                if "psver" in hf:
                    psver = int(np.asarray(hf["psver"]).ravel()[0])
        pipeline = UnwrapPipeline(patch_dir=patch_dir, psver=psver)
        pipeline.run()

    # ------------------------------------------------------------------
    # Step 7: Estimate SCLA + smooth
    # ------------------------------------------------------------------
    def run_step_7(self, patch_dir: Path) -> None:
        """Step 7 — Spatially-Correlated Look Angle (SCLA) estimation.

        MATLAB correspondence (stamps.m lines 509-526)
        ------------------------------------------------
        ```matlab
        if start_step<=7 & end_step >=7
            if strcmpi(small_baseline_flag,'y')
                ps_calc_scla(1,1)   % small baselines
                ps_smooth_scla(1)
                ps_calc_scla(0,1)   % single master
            else
                ps_calc_scla(0,1)
                ps_smooth_scla
            end
        end
        ```

        Estimates DEM error (K_ps_uw), master APS (C_ps_uw), and
        smooths by clipping outliers to Delaunay-neighbor range.
        Input: ps{v}.h5, bp{v}.h5, phuw{v}.h5; Output: scla{v}.h5, scla_smooth{v}.h5
        """
        psver = 2
        psver_path = patch_dir / "psver.h5"
        if psver_path.is_file():
            with h5py.File(str(psver_path), "r") as hf:
                if "psver" in hf:
                    psver = int(np.asarray(hf["psver"]).ravel()[0])
        estimator = SclaEstimator(patch_dir=patch_dir, psver=psver)
        estimator.run()

    # ------------------------------------------------------------------
    # Step 8: Filter spatially-correlated noise (SCN)
    # ------------------------------------------------------------------
    def run_step_8(self, patch_dir: Path) -> None:
        """Step 8 — Spatially-correlated noise (SCN) estimation.

        MATLAB correspondence (stamps.m lines 528-542)
        ------------------------------------------------
        ```matlab
        if start_step<=8 & end_step >=8
            if strcmpi(scn_kriging_flag,'y')
                ps_scn_filt_krig
            else
                ps_scn_filt
            end
        end
        ```

        Gaussian time + space low-pass on unwrapped phase (minus SCLA).
        Input: ps{v}.h5, phuw{v}.h5, scla{v}.h5; Output: scn{v}.h5
        """
        psver = 2
        psver_path = patch_dir / "psver.h5"
        if psver_path.is_file():
            with h5py.File(str(psver_path), "r") as hf:
                if "psver" in hf:
                    psver = int(np.asarray(hf["psver"]).ravel()[0])
        pipeline = ScnFiltPipeline(patch_dir=patch_dir, psver=psver)
        pipeline.run()

    # ------------------------------------------------------------------
    # Step 5b: Merge patches + IFG std (project-level, called ONCE)
    # ------------------------------------------------------------------
    def _run_step_5b(self) -> None:
        """Step 5b — Merge patches and compute per-IFG noise std.

        MATLAB correspondence (stamps.m lines 432-491)
        ------------------------------------------------
        ```matlab
        % Part 2: after all patches processed
        if stamps_PART2_flag == 'y'
            if patches_flag == 'y'
                ps_merge_patches
            end
            if abort_flag == 0
                ps_calc_ifg_std
            end
        end
        ```

        This method determines whether multi-patch merging is needed:
        - If ``patch.list`` exists with >= 1 patch directories → merge
        - Otherwise (single-patch / no patches) → skip merge

        Then ``calc_ifg_std`` is called at the project root.
        """
        logger.info(
            "\n======== Step 5b: Merge patches & IFG std [%s] ========",
            self.project_dir.name,
        )

        patches = _discover_patches(self.project_dir)
        patches_flag = (
            len(patches) > 0
            and not (len(patches) == 1 and patches[0] == self.project_dir)
        )

        abort_flag = False

        if patches_flag:
            # Multi-patch (or single PATCH_* subdir): run merge
            logger.info("Running ps_merge_patches (%d patches)", len(patches))
            merger = PatchMerger(project_dir=self.project_dir, psver=2)
            n_ps = merger.run()
            if n_ps == 0:
                abort_flag = True
        else:
            # Single-patch processing (project_dir == patch_dir)
            # Check if any step has no PS
            no_ps_path = self.project_dir / "no_ps_info.h5"
            if no_ps_path.is_file():
                stamps_step_no_ps = _load_no_ps_info(no_ps_path)
                if stamps_step_no_ps.sum() >= 1:
                    abort_flag = True

        if not abort_flag:
            calc_ifg_std(self.project_dir, psver=2)
        else:
            logger.info("No PS left — skipping ps_calc_ifg_std")


# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` CLI parser.

    Usage examples
    --------------
    ::

        # Run step 1 only
        python stamps_main.py --start 1 --end 1 --config path/to/test_data

        # Auto-resume from last completed step to step 4
        python stamps_main.py -s 0 -e 4 -c path/to/test_data
    """
    parser = argparse.ArgumentParser(
        prog="stamps_main",
        description=(
            "StaMPS/MTI Python port — persistent scatterer InSAR processing.\n"
            "Supports translated Steps 1-8, with later steps still under validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Step definitions:\n"
            "  1  Initial load of data\n"
            "  2  Estimate gamma\n"
            "  3  Select PS pixels\n"
            "  4  Weed out adjacent pixels\n"
            "  5  Correct wrapped phase & merge patches\n"
            "  6  Unwrap phase\n"
            "  7  Calc spatially-correlated DEM error\n"
            "  8  Filter spatially-correlated noise\n"
            "  0  Continue from last completed stage\n"
        ),
    )
    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=1,
        metavar="N",
        help="Starting step number (default: 1).",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=1,
        metavar="N",
        help="Ending step number (default: 1).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=".",
        metavar="DIR",
        help=(
            "Project directory containing parms.mat and PATCH_* subdirectories "
            "(default: current directory)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point called by the CLI or programmatically."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Logging setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    # Remove any pre-existing handlers (e.g. added by getparm.py on import)
    # to avoid duplicate output when root handler is also active.
    stamps_logger = logging.getLogger("stamps")
    for handler in stamps_logger.handlers[:]:
        stamps_logger.removeHandler(handler)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_dir = Path(args.config).resolve()
    if not project_dir.is_dir():
        logger.error("Project directory does not exist: %s", project_dir)
        return 1

    # Check that Python-native or legacy MATLAB parms are reachable.
    parm_candidates = [
        project_dir / "parms.json",
        project_dir / "parms.mat",
        project_dir.parent / "parms.json",
        project_dir.parent / "parms.mat",
    ]
    if not any(path.is_file() for path in parm_candidates):
        logger.error(
            "parms.json/parms.mat not found in %s or %s. "
            "Please specify a directory containing a parameter file via --config.",
            project_dir,
            project_dir.parent,
        )
        return 1

    try:
        runner = StampsRunner(project_dir=project_dir)
        runner.run(start_step=args.start, end_step=args.end)
    except FileNotFoundError as exc:
        logger.error("Missing required file: %s", exc)
        return 1
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("Unhandled error during processing")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
