"""
getparm.py — Python port of StaMPS getparm / ps_parms_default / setparm
========================================================================

Overview
--------
1. Provides a **singleton** class ``StampsConfig`` to manage all StaMPS
   processing parameters.
2. Implements module-level ``getparm(name)`` / ``setparm(name, value)``
   with the same behaviour as the MATLAB originals (prefix matching,
   localparms override priority, etc.).
3. Uses ``pathlib`` for cross-platform paths and ``scipy.io.loadmat``
   to read ``.mat`` files.
4. Uses Python ``logging`` instead of MATLAB ``logit``.

MATLAB → Python mapping
-----------------------
+-----------------------------+-----------------------------------------------+
| MATLAB concept              | Python mapping                                 |
+=============================+===============================================+
| ``setappdata(0,'parms',s)`` | ``StampsConfig`` singleton ``_parms`` dict     |
| ``getappdata(0,'parms')``   |                                               |
+-----------------------------+-----------------------------------------------+
| Global ``parms`` struct     | ``StampsConfig._parms: dict[str, Any]``        |
+-----------------------------+-----------------------------------------------+
| ``localparms`` struct       | ``StampsConfig._local_parms: dict[str, Any]``  |
+-----------------------------+-----------------------------------------------+
| ``parmfile = 'parms'``      | ``StampsConfig._parm_file: pathlib.Path``       |
| (+ implicit ``.mat``)       |                                               |
+-----------------------------+-----------------------------------------------+
| ``strmatch(prefix, names)`` | ``_prefix_match(prefix, names)`` helper        |
+-----------------------------+-----------------------------------------------+
| ``logit(msg)``              | ``logging.getLogger('stamps')``                |
+-----------------------------+-----------------------------------------------+

Numeric type conversion
-----------------------
MATLAB uses ``double`` (IEEE 754 64-bit) for scalars by default.
``scipy.io.loadmat`` returns MATLAB scalar doubles as ``numpy.float64``
and strings wrapped in ndarrays. This module normalises in
``_convert_mat_value``:
  - ``(1,1) ndarray of float64`` → ``numpy.float64``  (preserve precision)
  - ``(1,N) ndarray of float64`` → ``numpy.ndarray``  (flatten to 1D)
  - ``ndarray of str / bytes``   → ``str``
  - Empty ``(0,0) ndarray``      → ``[]`` (Python list)
so that Python values match MATLAB semantics and are convenient downstream.

Refactored from MATLAB StaMPS getparm.m / ps_parms_default.m / setparm.m
Original author: Andy Hooper, June 2006
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Logging — corresponds to MATLAB logit.m
# ---------------------------------------------------------------------------
# MATLAB logit writes to both STAMPS.log and stdout; here we use Python
# logging to console (and optionally add a file handler if needed).
logger = logging.getLogger("stamps")
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    logger.addHandler(_console)
    logger.setLevel(logging.INFO)


# ============================================================================
# Utility functions
# ============================================================================

def _convert_mat_value(val: Any) -> Any:
    """Convert MATLAB values returned by scipy.io.loadmat to Python types.

    MATLAB → Python type mapping
    -----------------------------
    - MATLAB ``double`` scalar (1×1 ndarray, dtype float64) → ``np.float64``
      (preserve float64 precision to match MATLAB default double).
    - MATLAB ``double`` row vector (1×N ndarray) → 1D ``np.ndarray``
    - MATLAB string → ``str``
    - MATLAB ``[]`` (0×0 ndarray) → ``[]`` (Python list)
    """
    if isinstance(val, np.ndarray):
        # --- strings ---
        if val.dtype.kind in ("U", "S", "O"):
            # MATLAB char array → Python str
            flat = val.flat
            if val.size == 1:
                s = str(flat[0])
                return s.strip()
            return str(val)
        # --- empty matrix ---
        if val.size == 0:
            return []
        # --- scalar ---
        if val.shape == (1, 1):
            # Keep np.float64 to match MATLAB double precision
            return val.flat[0]
        # --- row/column vector → 1D ---
        if val.ndim == 2 and (val.shape[0] == 1 or val.shape[1] == 1):
            return val.flatten()
        return val
    return val


def _json_to_value(val: Any) -> Any:
    """Convert JSON-friendly values to the runtime parameter representation."""
    if isinstance(val, list):
        if all(isinstance(item, (int, float, bool)) or item is None for item in val):
            return np.asarray([_json_to_value(item) for item in val])
        return [_json_to_value(item) for item in val]
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"inf", "+inf", "infinity", "+infinity"}:
            return np.float64(np.inf)
        if low in {"-inf", "-infinity"}:
            return np.float64(-np.inf)
        if low == "nan":
            return np.float64(np.nan)
    return val


def _value_to_json(val: Any) -> Any:
    """Convert numpy/MATLAB-style values to portable JSON values."""
    if isinstance(val, np.ndarray):
        return [_value_to_json(item) for item in val.tolist()]
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        f = float(val)
        if np.isnan(f):
            return "nan"
        if np.isposinf(f):
            return "inf"
        if np.isneginf(f):
            return "-inf"
        return f
    if isinstance(val, (list, tuple)):
        return [_value_to_json(item) for item in val]
    if isinstance(val, dict):
        return {str(k): _value_to_json(v) for k, v in sorted(val.items())}
    return val


def _prefix_match(prefix: str, names: List[str]) -> List[str]:
    """Emulate MATLAB ``strmatch(prefix, names)``: return names starting with *prefix*.

    MATLAB ``strmatch`` is prefix-based (not substring); same semantics here.
    """
    return [n for n in names if n.startswith(prefix)]


# ============================================================================
# Default parameter table — corresponds to ps_parms_default.m
# ============================================================================
# The dict below mirrors all if ~isfield(parms, ...) blocks in ps_parms_default.m.
# Value types match MATLAB where possible (float64 ↔ double, str ↔ char).
# Conditional defaults depending on small_baseline_flag are applied in _apply_defaults.

_STATIC_DEFAULTS: Dict[str, Any] = {
    # ---- Step 1: PS selection -------------------------------------------
    "max_topo_err": np.float64(20),
    "quick_est_gamma_flag": "y",
    "select_reest_gamma_flag": "y",
    "slc_osf": np.float64(1),           # SLC oversampling factor [MA]

    # ---- Filtering ------------------------------------------------------
    "filter_grid_size": np.float64(50),
    "filter_weighting": "P-square",
    "gamma_change_convergence": np.float64(0.005),
    "gamma_max_iterations": np.float64(3),

    # ---- CLAP filter ----------------------------------------------------
    "clap_win": np.float64(32),
    "clap_low_pass_wavelength": np.float64(800),
    "clap_alpha": np.float64(1),
    "clap_beta": np.float64(0.3),

    # ---- PS selection method --------------------------------------------
    "select_method": "DENSITY",
    "gamma_stdev_reject": np.float64(0),

    # ---- Weeding --------------------------------------------------------
    "weed_time_win": np.float64(730),
    "weed_max_noise": np.float64(np.inf),
    "weed_zero_elevation": "n",
    "weed_neighbours": "n",

    # ---- Unwrapping -----------------------------------------------------
    "unwrap_patch_phase": "n",
    "drop_ifg_index": [],
    "unwrap_la_error_flag": "y",
    "unwrap_spatial_cost_func_flag": "n",
    "unwrap_prefilter_flag": "y",
    "unwrap_grid_size": np.float64(200),
    "unwrap_gold_n_win": np.float64(32),
    "unwrap_alpha": np.float64(8),
    "unwrap_time_win": np.float64(730),
    "unwrap_gold_alpha": np.float64(0.8),
    "unwrap_hold_good_values": "n",

    # ---- SCLA -----------------------------------------------------------
    "scla_drop_index": [],
    "scn_wavelength": np.float64(100),
    "scn_time_win": np.float64(365),
    "scn_deramp_ifg": [],
    "scn_kriging_flag": "n",

    # ---- Reference area -------------------------------------------------
    "ref_lon": np.array([-np.inf, np.inf]),
    "ref_lat": np.array([-np.inf, np.inf]),
    "ref_centre_lonlat": np.array([0.0, 0.0]),
    "ref_radius": np.float64(np.inf),
    "ref_velocity": np.float64(0),

    # ---- Multi-core -----------------------------------------------------
    "n_cores": np.float64(1),

    # ---- Plotting -------------------------------------------------------
    "plot_dem_posting": np.float64(90),
    "plot_scatterer_size": np.float64(120),
    "plot_pixels_scatterer": np.float64(3),
    "plot_color_scheme": "inflation",
    "shade_rel_angle": np.array([90.0, 45.0]),
    "lonlat_offset": np.array([0.0, 0.0]),

    # ---- Merge ----------------------------------------------------------
    "merge_standard_dev": np.float64(np.inf),

    # ---- SCLA method ----------------------------------------------------
    "scla_method": "L2",
    "scla_deramp": "n",

    # ---- Troposphere ----------------------------------------------------
    "subtr_tropo": "n",
    "tropo_method": "a_l",
}

# Conditional defaults depending on small_baseline_flag
# Format: key → (sb_value, non_sb_value)
_CONDITIONAL_DEFAULTS: Dict[str, Tuple[Any, Any]] = {
    "density_rand":       (np.float64(2),     np.float64(20)),
    "percent_rand":       (np.float64(1),     np.float64(20)),
    "weed_standard_dev":  (np.float64(np.inf), np.float64(1.0)),
    "unwrap_method":      ("3D_QUICK",         "3D"),
    "merge_resample_size": (np.float64(100),   np.float64(0)),
}


# ============================================================================
# StampsConfig singleton class
# ============================================================================

class StampsConfig:
    """StaMPS parameter manager (thread-safe singleton).

    MATLAB correspondence
    --------------------
    In MATLAB StaMPS, parameters live in ``parms.mat`` as a struct, read/written
    globally via ``load`` / ``save``. ``getappdata(0, ...)`` and
    ``setappdata(0, ...)`` are also used in some steps to cache data on the
    MATLAB root application object.

    This class unifies that "global / implicit shared" behaviour as a
    **singleton + dict**:
    - ``_parms``       ↔ MATLAB ``parms`` struct (from parms.mat)
    - ``_local_parms`` ↔ MATLAB ``localparms`` struct (from localparms.mat)
    - Read priority: local_parms > parms > defaults
    """

    _instance: Optional["StampsConfig"] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton control
    # ------------------------------------------------------------------
    def __new__(cls, *args: Any, **kwargs: Any) -> "StampsConfig":
        """Thread-safe singleton creation.

        MATLAB: there is no singleton; ``parms.mat`` on disk acts as the single
        config source. In Python we use a singleton to avoid repeated loads and
        inconsistent state.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self, work_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialise config (only on first instance creation).

        Parameters
        ----------
        work_dir : str or Path, optional
            Working directory; default is current directory. MATLAB ``getparm``
            relies on ``pwd``; here we pass it explicitly for testability.
        """
        if self._initialized:
            return
        self._work_dir: Path = Path(work_dir) if work_dir else Path.cwd()
        self._parms: Dict[str, Any] = {}
        self._local_parms: Dict[str, Any] = {}
        self._parm_file: Optional[Path] = None
        self._local_parm_file: Optional[Path] = None
        self._parent_flag: bool = False  # True when parms file is in parent dir
        self._loaded: bool = False
        self._initialized = True

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------
    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance; next __new__ will create a new one. For tests only."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Load parms.json/parms.mat and local overrides
    # ------------------------------------------------------------------
    def load(self, work_dir: Optional[Union[str, Path]] = None) -> None:
        """Load StaMPS parameters from work_dir.

        Python-native JSON is preferred; MATLAB ``.mat`` files are kept as a
        compatibility fallback.

        Search order:
        1. ``./parms.json``
        2. ``./parms.mat``
        3. ``../parms.json``
        4. ``../parms.mat``

        Local overrides use the same rule in the working directory:
        ``localparms.json`` first, then ``localparms.mat``.

        MATLAB correspondence
        --------------------
        ```matlab
        if exist('./parms.mat','file')
            parms=load(parmfile);
        elseif exist('../parms.mat','file')
            parmfile='../parms';
            parms=load(parmfile);
        else
            error('parms.mat not found')
        end
        ```
        """
        if work_dir is not None:
            self._work_dir = Path(work_dir)

        wd = self._work_dir

        # ---------- Locate parms file ----------
        candidates = [
            (wd / "parms.json", False),
            (wd / "parms.mat", False),
            (wd.parent / "parms.json", True),
            (wd.parent / "parms.mat", True),
        ]
        for candidate, parent_flag in candidates:
            if candidate.is_file():
                self._parm_file = candidate
                self._parent_flag = parent_flag
                break
        else:
            raise FileNotFoundError(
                f"parms.json/parms.mat not found in {wd} or {wd.parent}"
            )

        logger.info("Loading parms from: %s", self._parm_file)
        data = self._load_parm_file_as_dict(self._parm_file)
        # If file has a single top-level variable that is a struct (dict), use it as parms
        if len(data) == 1 and isinstance(next(iter(data.values())), dict):
            self._parms = next(iter(data.values()))
        else:
            self._parms = data

        # ---------- Locate local overrides ----------
        local_path = None
        for candidate in [wd / "localparms.json", wd / "localparms.mat"]:
            if candidate.is_file():
                local_path = candidate
                break
        if local_path is not None:
            logger.info("Loading local parms from: %s", local_path)
            self._local_parm_file = local_path
            data_local = self._load_parm_file_as_dict(local_path)
            if len(data_local) == 1 and isinstance(next(iter(data_local.values())), dict):
                self._local_parms = next(iter(data_local.values()))
            else:
                self._local_parms = data_local
        else:
            self._local_parms = {}

        # ---------- Apply defaults ----------
        # Corresponds to full logic of MATLAB ps_parms_default.m
        self._apply_defaults()

        self._loaded = True
        logger.info(
            "Configuration loaded: %d parms, %d local overrides",
            len(self._parms),
            len(self._local_parms),
        )

    # ------------------------------------------------------------------
    # Read parameter files
    # ------------------------------------------------------------------
    @staticmethod
    def _load_parm_file_as_dict(path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return StampsConfig._load_json_as_dict(path)
        if suffix == ".mat":
            return StampsConfig._load_mat_as_dict(path)
        raise ValueError(f"Unsupported parameter file format: {path}")

    @staticmethod
    def _load_json_as_dict(json_path: Path) -> Dict[str, Any]:
        with json_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"{json_path} must contain a JSON object")
        return {str(k): _json_to_value(v) for k, v in raw.items()}

    @staticmethod
    def _load_mat_as_dict(mat_path: Path) -> Dict[str, Any]:
        """Load a .mat file and convert to a Python dict.

        Tries scipy.io.loadmat first (v4–v7). If the file is MATLAB v7.3 (HDF5),
        falls back to h5py. ndarrays are normalised via ``_convert_mat_value``
        (or equivalent for v7.3).
        """
        try:
            import scipy.io as sio
        except ImportError as exc:
            raise ImportError(
                "scipy is required to read .mat files. "
                "Install with: pip install scipy"
            ) from exc

        try:
            raw: dict = sio.loadmat(str(mat_path), squeeze_me=False)
        except Exception as exc:
            msg = str(exc).lower()
            if "v7.3" in msg or "hdf" in msg or "h5py" in msg:
                return StampsConfig._load_mat_v73_as_dict(mat_path)
            raise

        result: Dict[str, Any] = {}
        for key, val in raw.items():
            if key.startswith("__"):  # skip __header__, __version__, __globals__
                continue
            result[key] = _convert_mat_value(val)
        return result

    @staticmethod
    def _load_mat_v73_as_dict(mat_path: Path) -> Dict[str, Any]:
        """Load MATLAB v7.3 (.mat) file using h5py (HDF5).

        v7.3 files are HDF5; scipy.io.loadmat does not support them.
        MATLAB stores in column-major order; we transpose when reading.
        """
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "MATLAB v7.3 .mat files require h5py. "
                "Install with: pip install h5py"
            ) from exc

        def read_item(h5obj: "h5py.HDF5Object", file_root: "h5py.File") -> Any:
            if isinstance(h5obj, h5py.Group):
                out: Dict[str, Any] = {}
                for key in h5obj.keys():
                    if key == "#refs#":
                        continue
                    try:
                        out[key] = read_item(h5obj[key], file_root)
                    except Exception as e:
                        logger.debug("Skip %s/%s: %s", h5obj.name, key, e)
                return out
            if isinstance(h5obj, h5py.Dataset):
                d = h5obj[()]
                # MATLAB v7.3 often stores data via HDF5 object references
                if isinstance(d, np.ndarray) and d.dtype.kind == "O" and d.size > 0:
                    ref = d.flat[0]
                    try:
                        ref_obj = file_root[ref]
                        return read_item(ref_obj, file_root)
                    except (KeyError, TypeError, ValueError):
                        pass
                if isinstance(d, np.ndarray) and d.ndim >= 1:
                    d = np.transpose(d)
                if isinstance(d, np.ndarray) and d.dtype.names is not None:
                    names = d.dtype.names or ()
                    if "real" in names and "imag" in names:
                        d = d["real"] + 1j * d["imag"]
                    elif "r" in names and "i" in names:
                        d = d["r"] + 1j * d["i"]
                # MATLAB char array (uint16) → Python str
                if (
                    isinstance(d, np.ndarray)
                    and d.dtype.kind == "u"
                    and d.dtype.itemsize == 2
                    and d.size > 0
                ):
                    try:
                        return np.array(d).tobytes().decode("utf-16-le", errors="replace").strip()
                    except Exception:
                        pass
                return _convert_mat_value(d) if isinstance(d, np.ndarray) else d
            return None

        with h5py.File(str(mat_path), "r") as f:
            result = {}
            for key in f.keys():
                if key.startswith("#"):
                    continue
                try:
                    result[key] = read_item(f[key], f)
                except Exception as e:
                    logger.debug("Skip root key %s: %s", key, e)
            return result

    # ------------------------------------------------------------------
    # Apply default parameters (corresponds to ps_parms_default.m)
    # ------------------------------------------------------------------
    def _apply_defaults(self) -> None:
        """Fill in default values for missing parameters.

        Corresponds to all ``if ~isfield(parms, ...)`` blocks in ``ps_parms_default.m``.

        MATLAB correspondence
        ---------------------
        The MATLAB version checks each field and at the end ``save``s changes back
        to ``parms.mat``. Here we only update in memory; we do not write to file.

        Legacy field migration
        ----------------------
        ``ps_parms_default.m`` has ``rmfield`` logic to migrate old names
        (e.g. ``weed_alpha`` → ``weed_time_win``, ``plot_pixel_size`` →
        ``plot_scatterer_size``). We do the same.
        """
        parms = self._parms
        added: List[str] = []

        # --- Static defaults ---
        for key, default_val in _STATIC_DEFAULTS.items():
            if key not in parms:
                parms[key] = default_val
                added.append(key)

        # --- Conditional defaults (depend on small_baseline_flag) ---
        sb_flag = str(parms.get("small_baseline_flag", "n")).strip().lower()
        is_sb = sb_flag == "y"

        for key, (sb_val, non_sb_val) in _CONDITIONAL_DEFAULTS.items():
            if key not in parms:
                parms[key] = sb_val if is_sb else non_sb_val
                added.append(key)

        # --- sb_scla_drop_index: only in small_baseline mode ---
        # MATLAB ps_parms_default.m lines 387-391
        if is_sb and "sb_scla_drop_index" not in parms:
            parms["sb_scla_drop_index"] = []
            added.append("sb_scla_drop_index")

        # --- Legacy field migration ---
        # weed_alpha → weed_time_win (ps_parms_default.m lines 130-132)
        if "weed_alpha" in parms:
            parms.pop("weed_alpha")
            logger.debug("Removed deprecated field 'weed_alpha'")

        # plot_pixel_size → plot_scatterer_size (lines 301-304)
        if "plot_pixel_size" in parms:
            parms["plot_scatterer_size"] = np.float64(
                parms.pop("plot_pixel_size") * 25
            )
            logger.debug(
                "Migrated 'plot_pixel_size' → 'plot_scatterer_size'"
            )

        # pixel_aspect_ratio deprecated (lines 319-321)
        if "pixel_aspect_ratio" in parms:
            parms.pop("pixel_aspect_ratio")
            logger.debug("Removed deprecated field 'pixel_aspect_ratio'")

        # --- lambda / heading: try to load from text files ---
        # MATLAB ps_parms_default.m lines 351-381
        self._load_text_parm("lambda", "lambda.1.in", added)
        self._load_text_parm("heading", "heading.1.in", added)

        # --- insar_processor: read from processor.txt ---
        # MATLAB ps_parms_default.m lines 393-415
        if "insar_processor" not in parms:
            processor = self._read_processor_file()
            parms["insar_processor"] = processor
            added.append("insar_processor")

        if added:
            logger.info(
                "Applied %d default parameter(s): %s",
                len(added),
                ", ".join(added[:10]) + ("..." if len(added) > 10 else ""),
            )

    def _load_text_parm(
        self, parm_name: str, filename: str, added: List[str]
    ) -> None:
        """Try to load a single parameter (e.g. lambda, heading) from a text file.

        Search path: ``./ → ../ → ../../``, same as MATLAB.
        """
        if parm_name in self._parms:
            return

        wd = self._work_dir
        for candidate in [wd / filename, wd.parent / filename,
                          wd.parent.parent / filename]:
            if candidate.is_file():
                try:
                    val = np.loadtxt(str(candidate))
                    self._parms[parm_name] = (
                        np.float64(val) if val.ndim == 0 else val
                    )
                    logger.debug("Loaded %s from %s", parm_name, candidate)
                    added.append(parm_name)
                    return
                except Exception:
                    logger.warning(
                        "Failed to parse %s from %s", parm_name, candidate
                    )
                    break

        # File not found → NaN (same as MATLAB)
        self._parms[parm_name] = np.float64(np.nan)
        added.append(parm_name)

    def _read_processor_file(self) -> str:
        """Read processor.txt to determine InSAR processor type.

        MATLAB ps_parms_default.m lines 393-415: search upward for processor.txt;
        default is 'doris'.
        """
        wd = self._work_dir
        for candidate in [
            wd / "processor.txt",
            wd.parent / "processor.txt",
            wd.parent.parent / "processor.txt",
        ]:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text.lower() not in ("gamma", "doris", "isce"):
                    logger.warning(
                        "Unrecognised processor '%s' in %s "
                        "(supported: doris, gamma, isce)",
                        text,
                        candidate,
                    )
                return text
        return "doris"

    # ------------------------------------------------------------------
    # getparm — core read interface
    # ------------------------------------------------------------------
    def getparm(
        self,
        parm_name: Optional[str] = None,
        print_flag: bool = False,
    ) -> Any:
        """Get the value of a parameter.

        Logic matches MATLAB ``getparm.m``:

        1. If *parm_name* is ``None``, return **all parameters** as a sorted dict.
           (MATLAB: ``disp(orderfields(parms))``)
        2. Otherwise **prefix-match** (MATLAB ``strmatch``):
           - Single match → return value (localparms take precedence)
           - Multiple matches → raise ``ValueError``
           - No match → return ``None``

        Parameters
        ----------
        parm_name : str, optional
            Parameter name or a unique prefix.
        print_flag : bool
            If True, also log the parameter value (MATLAB ``printflag``).

        Returns
        -------
        value : Any
            Parameter value; ``None`` if no match.

        MATLAB correspondence
        ---------------------
        ```matlab
        parmnum=strmatch(parmname,fieldnames(parms));
        if length(parmnum)>1
            error(['Parameter ',parmname,'* is not unique'])
        elseif isempty(parmnum)
            parmname=[]; value=[];
        else
            ...
            if isfield(localparms,parmname)
                value=getfield(localparms,parmname);
            else
                value=getfield(parms,parmname);
            end
        end
        ```
        """
        if not self._loaded:
            self.load()

        # No argument → return all
        if parm_name is None:
            all_parms = dict(sorted(self._parms.items()))
            if self._local_parms:
                logger.info("--- Local parameter overrides ---")
                for k, v in sorted(self._local_parms.items()):
                    logger.info("  %s = %s", k, v)
            return all_parms

        # Exact match first, then prefix match
        # MATLAB strmatch returns all prefix matches, but if the exact name exists
        # in the parameter set, we should prefer it (avoids ambiguity when e.g.
        # 'lambda' matches both 'lambda' and 'lambda1').
        all_keys = list(self._parms.keys())
        if parm_name in all_keys:
            matches = [parm_name]
        else:
            matches = _prefix_match(parm_name, all_keys)
        if len(matches) > 1:
            raise ValueError(
                f"Parameter '{parm_name}*' is not unique. "
                f"Matches: {matches}"
            )
        if len(matches) == 0:
            logger.warning("Parameter '%s' not found", parm_name)
            return None

        resolved_name = matches[0]

        # localparms override parms (MATLAB getparm.m lines 54-58)
        if resolved_name in self._local_parms:
            value = self._local_parms[resolved_name]
        else:
            value = self._parms[resolved_name]

        # Optional logging (MATLAB getparm.m lines 60-67, calls logit)
        if print_flag:
            if isinstance(value, (int, float, np.integer, np.floating)):
                logger.info("%s = %g", resolved_name, value)
            elif isinstance(value, np.ndarray):
                logger.info("%s = %s", resolved_name, value)
            else:
                logger.info("%s = '%s'", resolved_name, value)

        return value

    # ------------------------------------------------------------------
    # setparm — write parameter interface
    # ------------------------------------------------------------------
    def setparm(
        self,
        parm_name: str,
        value: Any,
        new_flag: int = 0,
    ) -> None:
        """Set a parameter value.

        Corresponds to MATLAB ``setparm.m``:

        - ``new_flag == 0``  — update existing parameter (prefix match)
        - ``new_flag == 1``  — add new parameter to parms
        - ``new_flag == 2``  — add new parameter to localparms
        - ``new_flag == -1`` — remove parameter from parms
        - ``new_flag == -2`` — remove parameter from localparms

        If *value* is ``np.nan``, the parameter is reset to its default.

        MATLAB correspondence
        ---------------------
        ```matlab
        if isnan(value)
            parms=rmfield(parms,parmname);
            save(parmfile,'-struct','parms')
            ps_parms_default
            value=getparm(parmname);
        end
        ```
        """
        if not self._loaded:
            self.load()

        # --- Add new ---
        if new_flag == 1:
            self._parms[parm_name] = value
            logger.info("Added parm: %s = %s", parm_name, value)
            return
        if new_flag == 2:
            self._local_parms[parm_name] = value
            logger.info("Added LOCAL parm: %s = %s", parm_name, value)
            return

        # --- Prefix match ---
        matches = _prefix_match(parm_name, list(self._parms.keys()))
        if len(matches) > 1:
            raise ValueError(
                f"Parameter '{parm_name}*' is not unique. Matches: {matches}"
            )
        if len(matches) == 0:
            raise KeyError(f"Parameter '{parm_name}*' does not exist")
        resolved_name = matches[0]

        # --- Delete ---
        if new_flag == -1:
            self._parms.pop(resolved_name, None)
            logger.info("Removed parm: %s", resolved_name)
            return
        if new_flag == -2:
            self._local_parms.pop(resolved_name, None)
            logger.info("Removed LOCAL parm: %s", resolved_name)
            return

        # --- Reset to default (value == NaN) ---
        # MATLAB setparm.m lines 71-87
        if isinstance(value, float) and np.isnan(value):
            self._parms.pop(resolved_name, None)
            self._local_parms.pop(resolved_name, None)
            self._apply_defaults()
            new_val = self._parms.get(resolved_name)
            logger.info(
                "%s reset to default value: %s", resolved_name, new_val
            )
            return

        # --- Normal assignment ---
        # If in localparms update local, else update parms
        # (MATLAB setparm.m lines 179-186)
        if resolved_name in self._local_parms:
            self._local_parms[resolved_name] = value
            logger.info(
                "Updated LOCAL parm: %s = %s (Warning: only local updated)",
                resolved_name,
                value,
            )
        else:
            self._parms[resolved_name] = value
            logger.info("Updated parm: %s = %s", resolved_name, value)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def get_all_parms(self) -> Dict[str, Any]:
        """Return merged parameter dict (local overrides global)."""
        if not self._loaded:
            self.load()
        merged = dict(self._parms)
        merged.update(self._local_parms)
        return dict(sorted(merged.items()))

    def write_json(
        self,
        path: Union[str, Path],
        include_local: bool = False,
    ) -> None:
        """Write parameters to a Python-native JSON parameter file.

        Parameters
        ----------
        path
            Destination path, usually ``parms.json`` or ``localparms.json``.
        include_local
            When true, write the effective merged parameter set.  When false,
            write only the base ``parms`` dictionary.
        """
        if not self._loaded:
            self.load()
        data = self.get_all_parms() if include_local else dict(self._parms)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_value_to_json(data), fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        logger.info("Wrote JSON parms: %s", path)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        n_parms = len(self._parms) if self._loaded else 0
        return (
            f"<StampsConfig({status}, work_dir='{self._work_dir}', "
            f"parms={n_parms})>"
        )


def load_mat(mat_path: Union[str, Path], squeeze_me: bool = False) -> Dict[str, Any]:
    """Load a .mat file (v4–v7 or v7.3 HDF5). Use this instead of scipy.io.loadmat.

    Tries scipy.io.loadmat first; on v7.3/HDF error, falls back to h5py.
    Returns a dict mapping variable names to values (same as loadmat).
    Other modules should use this so that v7.3 .mat files work everywhere.
    """
    mat_path = Path(mat_path)
    try:
        import scipy.io as sio
        return sio.loadmat(str(mat_path), squeeze_me=squeeze_me)
    except Exception as exc:
        msg = str(exc).lower()
        if "v7.3" in msg or "hdf" in msg or "h5py" in msg:
            return StampsConfig._load_mat_v73_as_dict(mat_path)
        raise


# ============================================================================
# Module-level convenience functions — emulate MATLAB global getparm / setparm
# ============================================================================

def getparm(
    parm_name: Optional[str] = None,
    print_flag: bool = False,
    work_dir: Optional[Union[str, Path]] = None,
) -> Any:
    """Module-level ``getparm``; behaviour matches MATLAB ``getparm(parmname, printflag)``.

    On first use, the ``StampsConfig`` singleton is created and parameters are loaded.

    Parameters
    ----------
    parm_name : str, optional
        Parameter name or a unique prefix. If None, returns all parameters.
    print_flag : bool
        Whether to log the parameter value.
    work_dir : str or Path, optional
        Working directory (only used on first load; then use ``StampsConfig.reset()`` to change).
    """
    cfg = StampsConfig(work_dir=work_dir)
    if not cfg._loaded:
        cfg.load(work_dir)
    return cfg.getparm(parm_name, print_flag)


def setparm(
    parm_name: str,
    value: Any,
    new_flag: int = 0,
    work_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Module-level ``setparm``; behaviour matches MATLAB ``setparm(parmname, value, newflag)``."""
    cfg = StampsConfig(work_dir=work_dir)
    if not cfg._loaded:
        cfg.load(work_dir)
    cfg.setparm(parm_name, value, new_flag)


# ============================================================================
# __main__ — demo: load config from test_data/
# ============================================================================

if __name__ == "__main__":
    import sys

    # Resolve test_data path relative to this script
    script_dir = Path(__file__).resolve().parent
    test_data_dir = script_dir.parent / "test_data"

    if not test_data_dir.is_dir():
        print(f"ERROR: test_data directory not found at {test_data_dir}")
        sys.exit(1)

    # Reset singleton for a clean state
    StampsConfig.reset()

    # Create config instance and load
    config = StampsConfig(work_dir=test_data_dir)
    config.load()

    print("=" * 60)
    print(f"StaMPS Config loaded from: {test_data_dir}")
    print(f"Total parameters: {len(config._parms)}")
    print("=" * 60)

    # Demo: getparm — single parameter
    print("\n--- Demo: getparm('max_topo_err') ---")
    val = config.getparm("max_topo_err", print_flag=True)
    print(f"  Returned value: {val}  (type: {type(val).__name__})")

    # Demo: prefix match
    print("\n--- Demo: getparm('unwrap_gold_a') (prefix match) ---")
    val2 = config.getparm("unwrap_gold_a", print_flag=True)
    print(f"  Resolved to: unwrap_gold_alpha = {val2}")

    # Demo: string parameter
    print("\n--- Demo: getparm('small_baseline_flag') ---")
    sb = config.getparm("small_baseline_flag", print_flag=True)
    print(f"  Returned value: '{sb}'  (type: {type(sb).__name__})")

    # Demo: array parameter
    print("\n--- Demo: getparm('ref_lon') ---")
    ref = config.getparm("ref_lon", print_flag=True)
    print(f"  Returned value: {ref}  (type: {type(ref).__name__})")

    # Demo: all parameters
    print("\n--- Demo: getparm() — all parameters ---")
    all_p = config.getparm()
    for name, v in list(all_p.items())[:5]:
        print(f"  {name} = {v}")
    print(f"  ... ({len(all_p)} total)")

    # Demo: module-level function
    print("\n--- Demo: module-level getparm('filter_grid_size') ---")
    # Singleton already exists; no reset needed
    fgs = getparm("filter_grid_size", print_flag=True)
    print(f"  filter_grid_size = {fgs}")

    print("\nDone.")
