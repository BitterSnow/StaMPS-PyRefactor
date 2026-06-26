"""
data_loader.py — ISCE PS data loader (Python port of ps_load_initial_isce.m)
=============================================================================

Overview
--------
Provides ``ISCEPSLoader``, which replicates the single-master PS initial-load
pipeline from MATLAB ``ps_load_initial_isce.m``, reading ISCE-format
binary/text files and caching the results as ``.npz`` archives.

Only the **PS (Persistent Scatterer)** path is implemented here; all ``sb_*``
(Small Baseline) logic is excluded by design.

MATLAB → Python mapping
------------------------
+--------------------------------------------+---------------------------------------------+
| MATLAB concept                             | Python mapping                              |
+============================================+=============================================+
| ``fread(fid,[n*2,1],'float')``             | ``np.fromfile(fid, dtype=np.float32, ...)`` |
+--------------------------------------------+---------------------------------------------+
| ``complex(real, imag)``                    | ``real + 1j * imag`` → ``np.complex64``     |
+--------------------------------------------+---------------------------------------------+
| ``load(textfile)`` (whitespace-delimited)  | ``np.loadtxt(path)``                        |
+--------------------------------------------+---------------------------------------------+
| ``fread(fid,[2,inf],'float')'``            | ``np.fromfile(...).reshape(-1, 2)``         |
+--------------------------------------------+---------------------------------------------+
| ``datenum(y,m,d)``                         | ``datetime.date.toordinal() + 366``         |
+--------------------------------------------+---------------------------------------------+
| ``stamps_save('ps1', ...)``                | ``np.savez(...)``                           |
+--------------------------------------------+---------------------------------------------+
| ``setparm('heading', heading, 1)``         | ``StampsConfig.setparm(..., new_flag=1)``   |
+--------------------------------------------+---------------------------------------------+
| ``llh2local(llh, origin)``                 | ``ISCEPSLoader._llh2local(...)``            |
+--------------------------------------------+---------------------------------------------+

Binary file layout (ISCE conventions)
--------------------------------------
* **pscands.1.ph** — interleaved complex float32:
  Layout per interferogram column: ``[re0, im0, re1, im1, …, re_{n-1}, im_{n-1}]``
  where each element is a 4-byte IEEE 754 single-precision float (``np.float32``).
  Total bytes = ``n_ps × 2 × 4 × n_ifg``.

  MATLAB reads this with ``fread(fid, [n_ps*2, 1], 'float')`` per column.
  In Python we read ``2 * n_ps`` float32 values and view as complex64.

* **pscands.1.ll** — interleaved float32 lon/lat:
  ``[lon0, lat0, lon1, lat1, …]``  →  ``np.fromfile(..., np.float32).reshape(-1, 2)``

  MATLAB: ``fread(fid,[2,inf],'float')`` gives a ``(2, n_ps)`` matrix, transposed
  to ``(n_ps, 2)``.  In Python ``reshape(-1, 2)`` directly gives ``(n_ps, 2)``.

* **pscands.1.hgt** — flat float32, one height per PS:
  ``np.fromfile(..., np.float32)``

* **pscands.1.da** — flat float32, one dispersion value per PS:
  ``np.loadtxt(path)``  (MATLAB ``load(daname)`` treats it as text)

Width / Length offset note
--------------------------
MATLAB ``pscands.1.ij`` has 1-based ``[ID, Azimuth, Range]`` per row.
The Python port keeps the same array but all indexing into 2D grids
(e.g. incidence/look-angle grids) converts to 0-based with ``ij[:, col] - 1``
(MATLAB code does ``ij(:,3)+1`` because its own ij is already 0-based in the
stored file, while MATLAB arrays are 1-based).

Refactored from: ps_load_initial_isce.m, ps_load_initial.m, readcpx.m
Original author : Andy Hooper, June 2006
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import griddata

from getparm import StampsConfig

logger = logging.getLogger("stamps.loader")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# WGS84 ellipsoid constants used in llh2local (matches llh2local.m)
_WGS84_A = 6378137.0         # semi-major axis (m)
_WGS84_E = 0.08209443794970  # first eccentricity


# ============================================================================
# ISCEPSLoader
# ============================================================================

class ISCEPSLoader:
    """Load ISCE-format PS (single-master) data, mirroring ``ps_load_initial_isce.m``.

    The loader reads the standard ``pscands.1.*`` binary/text files produced by
    ``isce2stamps``, applies date-sorting, coordinate rotation, and outputs
    cached ``.npz`` archives for downstream StaMPS processing steps.

    Parameters
    ----------
    work_dir : str or Path
        Directory containing pscands files (typically a ``PATCH_*`` folder).
        Text metadata (``day.1.in``, ``bperp.1.in``, etc.) is searched first
        in *work_dir*, then in its parent ``../`` — same as the MATLAB version.
    psver : int
        PS version tag; defaults to 1 (matching MATLAB ``psver=1``).

    MATLAB correspondence
    ---------------------
    This class replaces the procedural script ``ps_load_initial_isce.m``.
    In MATLAB, results are saved as ``ps1.mat``, ``ph1.mat``, etc. via
    ``stamps_save``.  Here we use ``numpy.savez`` instead, writing
    ``ps1.npz``, ``ph1.npz``, etc.
    """

    # Standard ISCE file names (same order as ps_load_initial_isce.m lines 29-42)
    _FILE_NAMES = {
        "ph":         "pscands.1.ph",       # complex float32 phase per PS per ifg
        "ij":         "pscands.1.ij",       # [ID, Azimuth, Range] per PS (text, int)
        "bperp":      "bperp.1.in",         # perpendicular baseline per slave (text, float)
        "day":        "day.1.in",           # YYYYMMDD per slave (text, int)
        "master_day": "master_day.1.in",    # YYYYMMDD of master (text, int)
        "ll":         "pscands.1.ll",       # [lon, lat] per PS (binary float32)
        "da":         "pscands.1.da",       # dispersion index per PS (text, float)
        "hgt":        "pscands.1.hgt",      # height per PS (binary float32)
        "inc":        "pscands.1.inc",      # incidence angle per PS (binary float32, radians)
        "la_ps":      "pscands.1.la",       # look angle per PS (binary float32, radians)
        "la":         "look_angle.1.in",    # look angle grid (text)
        "heading":    "heading.1.in",       # satellite heading (text)
        "lambda":     "lambda.1.in",        # radar wavelength (text)
        "calamp":     "calamp.out",         # amplitude calibration (text)
        "width":      "width.txt",          # interferogram width in pixels (text)
        "len":        "len.txt",            # interferogram length in pixels (text)
    }

    def __init__(
        self,
        work_dir: Union[str, Path],
        psver: int = 1,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.psver = psver

        # ---- Outputs (populated by load()) ----
        self.day: Optional[np.ndarray] = None           # (n_image,) int32 ordinals
        self.master_day: Optional[int] = None            # int32 ordinal
        self.master_ix: Optional[int] = None             # 0-based index into day
        self.day_ix: Optional[np.ndarray] = None         # sort permutation of slaves
        self.bperp: Optional[np.ndarray] = None          # (n_image,) float64
        self.n_ifg: Optional[int] = None
        self.n_image: Optional[int] = None
        self.n_ps: Optional[int] = None
        self.ij: Optional[np.ndarray] = None             # (n_ps, 3) int32
        self.ph: Optional[np.ndarray] = None             # (n_ps, n_image) complex64
        self.lonlat: Optional[np.ndarray] = None         # (n_ps, 2) float64
        self.xy: Optional[np.ndarray] = None             # (n_ps, 3) float32 [id, x, y]
        self.sort_ix: Optional[np.ndarray] = None        # spatial sort permutation
        self.ll0: Optional[np.ndarray] = None            # (2,) lon/lat origin
        self.calconst: Optional[np.ndarray] = None       # (n_ifg-1,) float64
        self.hgt: Optional[np.ndarray] = None            # (n_ps,) float32
        self.inc: Optional[np.ndarray] = None            # (n_ps,) float32 radians
        self.la: Optional[np.ndarray] = None             # (n_ps,) float32 radians
        self.D_A: Optional[np.ndarray] = None            # (n_ps,) float32 or float64
        self.width: Optional[int] = None
        self.length: Optional[int] = None

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------
    def _resolve(self, key: str) -> Path:
        """Locate an ISCE file by *key*, searching work_dir then parent.

        MATLAB correspondence
        ---------------------
        Every MATLAB file access follows:
        ```matlab
        if ~exist(fname,'file')
            fname = ['../', fname];
        end
        ```
        We replicate this with pathlib.
        """
        name = self._FILE_NAMES[key]
        candidate = self.work_dir / name
        if candidate.is_file():
            return candidate
        parent_candidate = self.work_dir.parent / name
        if parent_candidate.is_file():
            return parent_candidate
        raise FileNotFoundError(
            f"{name} not found in {self.work_dir} or {self.work_dir.parent}"
        )

    def _resolve_optional(self, key: str) -> Optional[Path]:
        """Like ``_resolve`` but returns ``None`` instead of raising."""
        try:
            return self._resolve(key)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _yyyymmdd_to_ordinal(yyyymmdd: int) -> int:
        """Convert YYYYMMDD integer to a Python ordinal (days since 0001-01-01).

        MATLAB correspondence
        ---------------------
        MATLAB ``datenum(y,m,d)`` returns a serial date where 1 == 0000-Jan-01.
        Python ``datetime.date.toordinal()`` starts at 0001-Jan-01 = 1.
        MATLAB datenum for 0001-Jan-01 = 367, so::

            python_ordinal = matlab_datenum - 366

        We store Python ordinals throughout; conversion to MATLAB ordinals can
        be done with ``+ 366`` if comparing against .mat reference files.
        """
        y = yyyymmdd // 10000
        m = (yyyymmdd % 10000) // 100
        d = yyyymmdd % 100
        return datetime.date(y, m, d).toordinal()

    @staticmethod
    def _ordinal_to_datestr(ordinal: int, fmt: str = "%Y%m%d") -> str:
        """Convert ordinal back to a date string."""
        return datetime.date.fromordinal(ordinal).strftime(fmt)

    # ------------------------------------------------------------------
    # Coordinate conversion  (port of llh2local.m)
    # ------------------------------------------------------------------
    @staticmethod
    def _llh2local(lonlat: np.ndarray, origin: np.ndarray) -> np.ndarray:
        """Convert lon/lat to local XY (km), matching ``llh2local.m``.

        Parameters
        ----------
        lonlat : ndarray, shape (n, 2)
            Columns are [longitude, latitude] in **decimal degrees**.
        origin : ndarray, shape (2,)
            [lon0, lat0] reference in decimal degrees.

        Returns
        -------
        xy : ndarray, shape (n, 2)
            Local [x, y] in **kilometres** (east, north).

        MATLAB correspondence
        ---------------------
        ``llh2local.m`` expects ``(3, n)`` with rows [lon; lat; hgt] and returns
        ``(2, n)`` with rows [x; y].  We take ``(n, 2)`` and return ``(n, 2)``
        for more Pythonic column-major convention.

        The projection uses WGS84 ellipsoid constants and the transverse
        Mercator-like formulas from the original.
        """
        a = _WGS84_A
        e = _WGS84_E

        # Convert to radians (MATLAB: llh = double(llh) * pi / 180)
        lon = np.deg2rad(lonlat[:, 0].astype(np.float64))
        lat = np.deg2rad(lonlat[:, 1].astype(np.float64))
        lon0 = np.deg2rad(float(origin[0]))
        lat0 = np.deg2rad(float(origin[1]))

        xy = np.zeros((lonlat.shape[0], 2), dtype=np.float64)

        # Non-zero latitude (general case)
        nz = lat != 0.0

        if np.any(nz):
            dlambda = lon[nz] - lon0
            M = a * (
                (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat[nz]
                - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024)
                * np.sin(2 * lat[nz])
                + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat[nz])
                - (35 * e**6 / 3072) * np.sin(6 * lat[nz])
            )
            M0 = a * (
                (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat0
                - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024)
                * np.sin(2 * lat0)
                + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat0)
                - (35 * e**6 / 3072) * np.sin(6 * lat0)
            )
            N = a / np.sqrt(1 - e**2 * np.sin(lat[nz]) ** 2)
            E = dlambda * np.sin(lat[nz])

            xy[nz, 0] = N * np.cos(lat[nz]) / np.sin(lat[nz]) * np.sin(E)
            xy[nz, 1] = (
                M - M0
                + N * np.cos(lat[nz]) / np.sin(lat[nz]) * (1 - np.cos(E))
            )

        # Special case: latitude == 0
        z = ~nz
        if np.any(z):
            dlambda = lon[z] - lon0
            M0 = a * (
                (1 - e**2 / 4 - 3 * e**4 / 64 - 5 * e**6 / 256) * lat0
                - (3 * e**2 / 8 + 3 * e**4 / 32 + 45 * e**6 / 1024)
                * np.sin(2 * lat0)
                + (15 * e**4 / 256 + 45 * e**6 / 1024) * np.sin(4 * lat0)
                - (35 * e**6 / 3072) * np.sin(6 * lat0)
            )
            xy[z, 0] = a * dlambda
            xy[z, 1] = -M0

        # Convert m → km
        xy /= 1000.0
        return xy

    # ------------------------------------------------------------------
    # Phase reading  (port of the fread loop + readcpx.m)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_complex_phase(
        path: Path, n_ps: int, n_cols: int
    ) -> np.ndarray:
        """Read interleaved complex float32 phase file.

        Returns shape ``(n_ps, n_cols)`` with dtype ``complex64``.

        Binary layout (per column / interferogram)
        -------------------------------------------
        ``[re_0, im_0, re_1, im_1, …, re_{n-1}, im_{n-1}]``
        — total ``n_ps × 2`` float32 values per column.

        MATLAB correspondence (ps_load_initial_isce.m lines 136-143)
        -------------------------------------------------------------
        ```matlab
        for i = 1 : n_ifg-1
            [ph_bit, byte_count] = fread(fid, [n_ps*2, 1], 'float');
            ph_bit = single(ph_bit);
            ph(:,i) = complex(ph_bit(1:2:end), ph_bit(2:2:end));
        end
        ```

        In Python we read the entire file at once for efficiency and
        then ``view`` as complex64 (re-interpreting pairs of float32 as
        one complex64, zero-copy):

        ``raw.reshape(n_cols, n_ps, 2)``  →  view as complex64
        →  squeeze last dim  → transpose to ``(n_ps, n_cols)``.

        ISCE data is **single-precision** (``float32``); ``complex64`` in
        NumPy is two float32 components — matching MATLAB ``single`` complex.
        """
        expected_bytes = n_ps * 2 * 4 * n_cols
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{path.name} has {actual_bytes} bytes "
                f"(expected {expected_bytes} = {n_ps}×2×4×{n_cols})"
            )

        # Read entire file as float32, then re-interpret as complex64
        raw = np.fromfile(str(path), dtype=np.float32)
        # Reshape to (n_cols, n_ps, 2) — column-major in MATLAB corresponds
        # to consecutive n_ps*2 floats per column.
        raw = raw.reshape(n_cols, n_ps, 2)
        # View pairs of float32 as complex64 (zero-copy reinterpretation)
        ph = raw[:, :, 0] + 1j * raw[:, :, 1]
        ph = ph.astype(np.complex64)
        # Transpose to (n_ps, n_cols)
        return ph.T

    # ------------------------------------------------------------------
    # Lon/lat reading  (binary float32)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_lonlat(path: Path, n_ps: int) -> np.ndarray:
        """Read binary lon/lat file.  Returns ``(n_ps, 2)`` float32.

        Binary layout
        -------------
        ``[lon_0, lat_0, lon_1, lat_1, …]``
        — ``n_ps × 2`` float32 values, total ``n_ps × 8`` bytes.

        MATLAB correspondence (ps_load_initial_isce.m lines 153-156)
        -------------------------------------------------------------
        ```matlab
        fid = fopen(llname, 'r');
        lonlat = fread(fid, [2, inf], 'float');
        lonlat = lonlat';
        ```
        ``fread(…, [2, inf], 'float')`` reads column-by-column, producing a
        ``(2, n_ps)`` matrix.  The transpose gives ``(n_ps, 2)``.

        In Python, ``np.fromfile`` reads in row-major order, so
        ``reshape(-1, 2)`` directly yields ``(n_ps, 2)`` — the lon/lat pairs
        are stored consecutively: ``[lon0, lat0, lon1, lat1, …]``.
        """
        data = np.fromfile(str(path), dtype=np.float32)
        if data.size != n_ps * 2:
            raise ValueError(
                f"{path.name} has {data.size // 2} entries, expected {n_ps}"
            )
        return data.reshape(-1, 2)

    # ------------------------------------------------------------------
    # Height reading  (binary float32)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_hgt(path: Path) -> np.ndarray:
        """Read height file.  Returns 1D float32 array.

        MATLAB (lines 211-215):
        ```matlab
        fid = fopen(hgtname, 'r');
        hgt = fread(fid, [1, inf], 'float');
        hgt = hgt';
        ```
        """
        return np.fromfile(str(path), dtype=np.float32)

    # ------------------------------------------------------------------
    # ISCE float raster  (MATLAB load_isce for baseline / incidence grids)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_isce_raster_file(stem: Path) -> Path:
        """Resolve *stem* to a readable float raster file (file or directory)."""
        if stem.is_file():
            return stem
        if stem.is_dir():
            for pattern in ("*.r4", "*.flt", "*.float", "*.bin"):
                hits = list(stem.glob(pattern))
                if hits:
                    return max(hits, key=lambda p: p.stat().st_size)
            any_files = [p for p in stem.iterdir() if p.is_file()]
            if any_files:
                return max(any_files, key=lambda p: p.stat().st_size)
        for candidate in (stem.with_suffix(".r4"), stem.with_suffix(".flt")):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"No ISCE raster file found for {stem}")

    @classmethod
    def _load_isce_float_raster(
        cls,
        stem: Path,
        full_w: Optional[int],
        full_l: Optional[int],
    ) -> np.ndarray:
        """Load a 2-D float32 grid like MATLAB ``load_isce``.

        If *full_w* and *full_l* are given and ``size == full_w * full_l``, the
        array is reshaped to ``(full_w, full_l)`` in C order (row = range /
        first index, column = azimuth / second), matching ``size(bperp_grid)==[width,len]``.

        Otherwise the flat length is factored into a pair ``(nrow, ncol)`` with
        aspect ratio closest to ``full_w : full_l`` (when known).
        """
        path = cls._resolve_isce_raster_file(stem)
        try:
            data = np.fromfile(str(path), dtype=np.float32)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read raster bytes: {path}") from exc
        if data.size == 0:
            try:
                data = np.loadtxt(str(path), dtype=np.float32)
            except Exception as exc:
                raise ValueError(f"Empty or unreadable raster: {path}") from exc

        n_el = int(data.size)
        if full_w is not None and full_l is not None and n_el == full_w * full_l:
            return data.reshape(full_w, full_l)

        target_ratio = (
            (full_w / full_l) if full_w and full_l and full_l > 0 else 1.0
        )
        best: Optional[Tuple[float, int, int]] = None
        for nrow in range(1, n_el + 1):
            if n_el % nrow != 0:
                continue
            ncol = n_el // nrow
            score = abs((nrow / ncol) - target_ratio) if ncol else np.inf
            if best is None or score < best[0]:
                best = (score, nrow, ncol)
        if best is None:
            raise ValueError(
                f"Cannot infer 2-D shape for raster {path} ({n_el} elements)"
            )
        _, nrow, ncol = best
        return data.reshape(nrow, ncol)

    @staticmethod
    def _interpolate_bperp_grid(
        bperp_grid: np.ndarray,
        ij_range: np.ndarray,
        ij_az: np.ndarray,
        full_w: int,
        full_l: int,
    ) -> np.ndarray:
        """Linear interpolation on a decimated grid (MATLAB ``griddata`` branch)."""
        nrow, ncol = bperp_grid.shape
        x_line = np.linspace(0.0, float(full_w), nrow, dtype=np.float64)
        y_line = np.linspace(0.0, float(full_l), ncol, dtype=np.float64)
        rr, cc = np.meshgrid(
            np.arange(nrow, dtype=np.int32),
            np.arange(ncol, dtype=np.int32),
            indexing="ij",
        )
        points = np.column_stack([x_line[rr.ravel()], y_line[cc.ravel()]])
        values = bperp_grid[rr, cc].ravel()
        xi = np.column_stack(
            [ij_range.astype(np.float64), ij_az.astype(np.float64)]
        )
        return griddata(points, values, xi, method="linear").astype(np.float32)

    def _find_baseline_grid_parent(self) -> Optional[Path]:
        """Parent directory that contains ``baselineGRID_*`` entries (MATLAB lines 283-287)."""
        for base in (self.work_dir.parent, self.work_dir.parent.parent):
            if any(base.glob("baselineGRID_*")):
                return base
        return None

    # ------------------------------------------------------------------
    # calamp.out parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_calamp(
        path: Path, master_day_yyyymmdd: int
    ) -> np.ndarray:
        """Parse amplitude calibration file ``calamp.out``.

        Returns sorted calibration constants (excluding master).

        MATLAB correspondence (ps_load_initial_isce.m lines 107-134)
        -------------------------------------------------------------
        Each line has ``<slc_path>  <cal_constant>``.
        The date is extracted from the path string (8-digit YYYYMMDD).
        Master entries are dropped, then sorted by date.

        The MATLAB code attempts several path-parsing heuristics (last
        component, second-to-last, etc.) to extract the date.  We replicate
        the same logic.
        """
        calfiles: List[str] = []
        calconsts: List[float] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                calfiles.append(parts[0])
                calconsts.append(float(parts[1]))

        cal_arr = np.array(calconsts, dtype=np.float64)
        cal_dates = np.zeros(len(calfiles), dtype=np.int64)

        for i, fpath in enumerate(calfiles):
            # Split by '/' to mimic MATLAB strread with delimiter '/'
            tokens = fpath.replace("\\", "/").split("/")
            date_val = None
            # Try last token first 8 chars
            try:
                date_val = int(tokens[-1][:8])
            except (ValueError, IndexError):
                pass
            if date_val is None or date_val < 19000000 or date_val > 20990000:
                # Try second-to-last token
                try:
                    date_val = int(tokens[-2][:8])
                except (ValueError, IndexError):
                    pass
            if date_val is None or date_val < 19000000 or date_val > 20990000:
                # Try third-to-last token last 8 chars (for 'master' paths)
                try:
                    date_val = int(tokens[-3][-8:])
                except (ValueError, IndexError):
                    pass
            if date_val is not None:
                cal_dates[i] = date_val

        # Remove master entries
        not_master = cal_dates != master_day_yyyymmdd
        cal_dates = cal_dates[not_master]
        cal_arr = cal_arr[not_master]

        # Sort by date
        sort_idx = np.argsort(cal_dates)
        return cal_arr[sort_idx]

    # ==================================================================
    # Main load pipeline
    # ==================================================================
    def load(self) -> "ISCEPSLoader":
        """Execute the full PS initial-load pipeline.

        This method mirrors the sequential logic of ``ps_load_initial_isce.m``
        from top to bottom.  Each section is annotated with the corresponding
        MATLAB line numbers.

        Returns *self* for method chaining.
        """
        logger.info("Loading ISCE PS data from: %s", self.work_dir)

        # ==============================================================
        # 1. Dates  (MATLAB lines 52-73)
        # ==============================================================
        # day.1.in — one YYYYMMDD per slave SLC
        day_path = self._resolve("day")
        day_raw = np.loadtxt(str(day_path), dtype=np.int64)
        slave_ordinals = np.array(
            [self._yyyymmdd_to_ordinal(int(d)) for d in day_raw],
            dtype=np.int32,
        )
        # Sort slave dates (MATLAB: [slave_day, day_ix] = sort(slave_day))
        self.day_ix = np.argsort(slave_ordinals).astype(np.int32)
        slave_ordinals = slave_ordinals[self.day_ix]

        # master_day.1.in
        master_path = self._resolve("master_day")
        master_yyyymmdd = int(np.loadtxt(str(master_path)))
        master_ord = self._yyyymmdd_to_ordinal(master_yyyymmdd)
        self.master_day = np.int32(master_ord)

        # Insert master into sorted slave list
        # MATLAB: master_ix = sum(slave_day < master_day) + 1   (1-based)
        # Python: 0-based index
        master_ix_0 = int(np.sum(slave_ordinals < master_ord))
        self.master_ix = np.int32(master_ix_0)

        # MATLAB: day = [slave(1:mix-1); master; slave(mix:end)]
        self.day = np.insert(slave_ordinals, master_ix_0, master_ord).astype(
            np.int32
        )
        logger.info(
            "Dates loaded: %d slaves + 1 master → %d images, master_ix=%d",
            len(slave_ordinals),
            len(self.day),
            self.master_ix,
        )

        # ==============================================================
        # 2. Baselines  (MATLAB lines 77-84)
        # ==============================================================
        bperp_path = self._resolve("bperp")
        bperp_raw = np.loadtxt(str(bperp_path), dtype=np.float64)
        bperp_sorted = bperp_raw[self.day_ix]
        # Insert 0 for master-master baseline
        self.bperp = np.insert(bperp_sorted, master_ix_0, 0.0)
        self.n_ifg = len(self.bperp)
        self.n_image = self.n_ifg  # PS mode: n_image == n_ifg
        logger.info("Baselines loaded: n_ifg=%d", self.n_ifg)

        # ==============================================================
        # 3. Heading & wavelength  (MATLAB lines 87-101)
        # ==============================================================
        heading_path = self._resolve("heading")
        heading = float(np.loadtxt(str(heading_path)))
        if heading == 0.0:
            raise ValueError("heading.1.in is empty or zero")
        # Store in StampsConfig (equivalent to setparm('heading', heading, 1))
        try:
            cfg = StampsConfig()
            if cfg._loaded:
                cfg.setparm("heading", heading, new_flag=1)
        except Exception:
            pass  # config not yet loaded; will be set later
        logger.info("Heading: %.6f deg", heading)

        lambda_path = self._resolve("lambda")
        wavelength = float(np.loadtxt(str(lambda_path)))
        try:
            cfg = StampsConfig()
            if cfg._loaded:
                cfg.setparm("lambda", wavelength, new_flag=1)
        except Exception:
            pass
        logger.info("Wavelength: %.4f m", wavelength)

        # ==============================================================
        # 4. Radar coordinates  (MATLAB line 104-105)
        # ==============================================================
        ij_path = self._resolve("ij")
        ij = np.loadtxt(str(ij_path), dtype=np.int32)
        self.n_ps = ij.shape[0]
        logger.info("PS candidates: n_ps=%d", self.n_ps)

        # ==============================================================
        # 5. Amplitude calibration  (MATLAB lines 107-134)
        # ==============================================================
        calamp_path = self._resolve_optional("calamp")
        if calamp_path is not None:
            self.calconst = self._parse_calamp(calamp_path, master_yyyymmdd)
            # Ensure length matches n_ifg - 1 (number of slave images)
            if len(self.calconst) != self.n_ifg - 1:
                logger.warning(
                    "calamp.out produced %d constants, expected %d; "
                    "falling back to ones",
                    len(self.calconst),
                    self.n_ifg - 1,
                )
                self.calconst = np.ones(self.n_ifg - 1, dtype=np.float64)
        else:
            self.calconst = np.ones(self.n_ifg - 1, dtype=np.float64)
        logger.info(
            "Calibration constants: %d values loaded", len(self.calconst)
        )

        # ==============================================================
        # 6. Complex phase  (MATLAB lines 136-149)
        # ==============================================================
        # MATLAB reads n_ifg-1 columns (one per slave), each with n_ps complex values.
        # Binary layout: n_ps*2 float32 per column, interleaved [re,im,...].
        ph_path = self._resolve("ph")
        n_slave_ifgs = self.n_ifg - 1  # expected columns in ph file

        # Detect actual column count from file size
        actual_bytes = ph_path.stat().st_size
        actual_cols = actual_bytes // (self.n_ps * 8)
        if actual_cols != n_slave_ifgs:
            logger.warning(
                "Phase file has %d columns (file_size / n_ps / 8), "
                "expected %d for PS mode.  Loading actual columns.",
                actual_cols,
                n_slave_ifgs,
            )
            n_slave_ifgs = actual_cols

        ph = self._read_complex_phase(ph_path, self.n_ps, n_slave_ifgs)

        # Reorder columns by day_ix (sort slaves by date)
        # MATLAB line 145: ph = ph(:, day_ix)
        if n_slave_ifgs == self.n_ifg - 1:
            ph = ph[:, self.day_ix]

        # MATLAB line 146-147: drop PS where more than 1 phase is zero
        zero_ph = np.sum(ph == 0, axis=1)
        nonzero_ix = zero_ph <= 1

        # Scale by calibration constants (MATLAB line 148)
        # ph = ph ./ repmat(calconst', n_ps, 1)
        if n_slave_ifgs == len(self.calconst):
            ph = ph / self.calconst[np.newaxis, :]

        # Insert master column (ones) at master_ix position
        # MATLAB line 149: ph = [ph(:,1:mix-1), ones(n_ps,1), ph(:,mix:end)]
        master_col = np.ones((self.n_ps, 1), dtype=np.complex64)
        self.ph = np.hstack([
            ph[:, :master_ix_0], master_col, ph[:, master_ix_0:]
        ]).astype(np.complex64)
        logger.info("Phase loaded: shape=%s, dtype=%s", self.ph.shape, self.ph.dtype)

        # ==============================================================
        # 7. Lon/Lat → local XY  (MATLAB lines 153-188)
        # ==============================================================
        ll_path = self._resolve("ll")
        lonlat = self._read_lonlat(ll_path, self.n_ps).astype(np.float64)

        # ll0 = centre of extent (MATLAB line 158)
        self.ll0 = (lonlat.max(axis=0) + lonlat.min(axis=0)) / 2.0

        # Convert to local km, then to metres
        # MATLAB: xy = llh2local(lonlat', ll0) * 1000
        xy = self._llh2local(lonlat, self.ll0) * 1000.0  # (n_ps, 2) in metres

        # ---- Heading-based rotation (MATLAB lines 169-181) ----
        theta = np.deg2rad(180.0 - heading)
        if theta > np.pi:
            theta -= 2 * np.pi

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotm = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        xy_T = xy.T                 # (2, n_ps)
        xy_new = rotm @ xy_T        # rotated

        # Only keep rotation if it reduces bounding-box span in both dims
        if (
            np.ptp(xy_new[0]) < np.ptp(xy_T[0])
            and np.ptp(xy_new[1]) < np.ptp(xy_T[1])
        ):
            xy_T = xy_new
            logger.info("Rotating coordinates by %.2f degrees", np.rad2deg(theta))

        xy = xy_T.T.astype(np.float32)  # back to (n_ps, 2)

        # Sort by ascending y then x (MATLAB line 184)
        sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
        xy = xy[sort_ix]

        # Prepend 1-based ID column, round to mm (MATLAB lines 186-187)
        ids = np.arange(1, self.n_ps + 1, dtype=np.float32).reshape(-1, 1)
        xy_rounded = np.round(xy * 1000) / 1000.0
        xy_full = np.hstack([ids, xy_rounded]).astype(np.float32)

        self.sort_ix = sort_ix.astype(np.int32)

        # ==============================================================
        # 8. Reorder everything by sort_ix  (MATLAB lines 189-192)
        # ==============================================================
        self.ph = self.ph[sort_ix]
        self.ij = ij[sort_ix]
        self.ij[:, 0] = np.arange(1, self.n_ps + 1)  # re-number IDs
        self.lonlat = lonlat[sort_ix]
        self.xy = xy_full

        # ==============================================================
        # 9. Dispersion (D_A)  (MATLAB lines 202-207)
        # ==============================================================
        da_path = self._resolve_optional("da")
        if da_path is not None:
            try:
                D_A = np.loadtxt(str(da_path), dtype=np.float64)
                if D_A.shape[0] == self.n_ps:
                    self.D_A = D_A[sort_ix]
                else:
                    logger.warning(
                        "D_A has %d entries, n_ps=%d — skipping",
                        D_A.shape[0],
                        self.n_ps,
                    )
            except Exception as exc:
                logger.warning("Failed to load D_A: %s", exc)

        # ==============================================================
        # 10. Height  (MATLAB lines 210-218)
        # ==============================================================
        hgt_path = self._resolve_optional("hgt")
        if hgt_path is not None:
            hgt = self._read_hgt(hgt_path)
            if hgt.shape[0] == self.n_ps:
                self.hgt = hgt[sort_ix]
            else:
                logger.warning(
                    "hgt has %d entries, n_ps=%d — skipping",
                    hgt.shape[0],
                    self.n_ps,
                )

        for key, attr_name in [("inc", "inc"), ("la_ps", "la")]:
            angle_path = self._resolve_optional(key)
            if angle_path is None:
                continue
            angle = self._read_hgt(angle_path)
            if angle.shape[0] == self.n_ps:
                setattr(self, attr_name, angle[sort_ix])
            else:
                logger.warning(
                    "%s has %d entries, n_ps=%d — skipping",
                    angle_path.name,
                    angle.shape[0],
                    self.n_ps,
                )

        # ==============================================================
        # 11. Width & Length  (MATLAB lines 220-228)
        # ==============================================================
        try:
            width_path = self._resolve("width")
            self.width = int(np.loadtxt(str(width_path)))
        except FileNotFoundError:
            self.width = None

        try:
            len_path = self._resolve("len")
            self.length = int(np.loadtxt(str(len_path)))
        except FileNotFoundError:
            self.length = None

        # ==============================================================
        # 12. Per-PS baselines (bperp_mat)  (MATLAB lines 283-324)
        # ==============================================================
        # Either load one spatial grid per slave from ``baselineGRID_YYYYMMDD``
        # (under ``../`` or ``../../`` relative to the patch) or replicate scalar
        # bperp.  Sample grids using *unsorted* ``ij`` (file order), then permute
        # rows with ``sort_ix`` to match sorted ``self.ij`` / ``self.ph``.
        bperp_no_master = np.delete(self.bperp, master_ix_0).astype(np.float32)
        grid_parent = self._find_baseline_grid_parent()
        image_ix_order = [
            i for i in range(self.n_image) if i != master_ix_0
        ]
        use_scalar_bperp = True
        if (
            grid_parent is not None
            and self.width is not None
            and self.length is not None
        ):
            bperp_fill = np.zeros(
                (self.n_ps, self.n_image - 1), dtype=np.float32
            )
            failed = False
            for col_ix, img_ix in enumerate(image_ix_order):
                ymd = self._ordinal_to_datestr(int(self.day[img_ix]))
                stem = grid_parent / f"baselineGRID_{ymd}"
                try:
                    bperp_grid = self._load_isce_float_raster(
                        stem, self.width, self.length
                    ).astype(np.float32)
                except (FileNotFoundError, ValueError, OSError) as exc:
                    logger.warning(
                        "Baseline grid load failed for %s: %s — using "
                        "scalar bperp fallback for all PS",
                        stem,
                        exc,
                    )
                    failed = True
                    break
                ij_rng = ij[:, 2]
                ij_az = ij[:, 1]
                if bperp_grid.shape == (
                    int(self.width),
                    int(self.length),
                ):
                    bperp_fill[:, col_ix] = bperp_grid[ij_rng, ij_az]
                else:
                    bperp_fill[:, col_ix] = self._interpolate_bperp_grid(
                        bperp_grid,
                        ij_rng,
                        ij_az,
                        int(self.width),
                        int(self.length),
                    )
            if not failed:
                self.bperp_mat = bperp_fill[self.sort_ix, :].astype(np.float32)
                use_scalar_bperp = False
                logger.info(
                    "bperp_mat: shape=%s (per-slave baseline grids)",
                    self.bperp_mat.shape,
                )

        if use_scalar_bperp:
            self.bperp_mat = np.tile(bperp_no_master, (self.n_ps, 1))
            logger.info(
                "bperp_mat: shape=%s (scalar baselines replicated)",
                self.bperp_mat.shape,
            )

        logger.info("ISCE PS data loading complete.")
        return self

    # ------------------------------------------------------------------
    # Save / cache  (port of stamps_save calls)
    # ------------------------------------------------------------------
    def save(self, out_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Serialise loaded data to ``.npz`` files.

        Produces the same set of cache files as the MATLAB ``stamps_save`` calls:
        - ``ps{ver}.npz``  — ij, lonlat, xy, bperp, day, master_day, …
        - ``ph{ver}.npz``  — ph (complex phase matrix)
        - ``da{ver}.npz``  — D_A (dispersion)
        - ``hgt{ver}.npz`` — hgt (height)
        - ``bp{ver}.npz``  — bperp_mat (per-PS baselines)
        - ``psver.npz``    — psver scalar

        Parameters
        ----------
        out_dir : str or Path, optional
            Output directory; defaults to *work_dir*.

        Returns
        -------
        dict mapping name → Path of each written file.
        """
        if out_dir is None:
            out_dir = self.work_dir
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ver = str(self.psver)
        written: Dict[str, Path] = {}

        # ps{ver}.npz — main PS info
        ps_path = out_dir / f"ps{ver}.npz"
        np.savez(
            str(ps_path),
            ij=self.ij,
            lonlat=self.lonlat,
            xy=self.xy,
            bperp=self.bperp,
            day=self.day,
            master_day=self.master_day,
            master_ix=self.master_ix,
            n_ifg=np.int32(self.n_ifg),
            n_image=np.int32(self.n_image),
            n_ps=np.int32(self.n_ps),
            sort_ix=self.sort_ix,
            ll0=self.ll0,
            calconst=self.calconst,
            day_ix=self.day_ix,
        )
        written["ps"] = ps_path
        logger.info("Saved %s", ps_path)

        # ph{ver}.npz — complex phase
        ph_path = out_dir / f"ph{ver}.npz"
        np.savez(str(ph_path), ph=self.ph)
        written["ph"] = ph_path
        logger.info("Saved %s", ph_path)

        # da{ver}.npz — dispersion
        if self.D_A is not None:
            da_path = out_dir / f"da{ver}.npz"
            np.savez(str(da_path), D_A=self.D_A)
            written["da"] = da_path
            logger.info("Saved %s", da_path)

        # hgt{ver}.npz — height
        if self.hgt is not None:
            hgt_path = out_dir / f"hgt{ver}.npz"
            np.savez(str(hgt_path), hgt=self.hgt)
            written["hgt"] = hgt_path
            logger.info("Saved %s", hgt_path)

        if self.inc is not None:
            inc_path = out_dir / f"inc{ver}.npz"
            np.savez(str(inc_path), inc=self.inc)
            written["inc"] = inc_path
            logger.info("Saved %s", inc_path)

        if self.la is not None:
            la_path = out_dir / f"la{ver}.npz"
            np.savez(str(la_path), la=self.la)
            written["la"] = la_path
            logger.info("Saved %s", la_path)

        # bp{ver}.npz — per-PS baselines
        if self.bperp_mat is not None:
            bp_path = out_dir / f"bp{ver}.npz"
            np.savez(str(bp_path), bperp_mat=self.bperp_mat)
            written["bp"] = bp_path
            logger.info("Saved %s", bp_path)

        # psver.npz
        psver_path = out_dir / "psver.npz"
        np.savez(str(psver_path), psver=np.int32(self.psver))
        written["psver"] = psver_path
        logger.info("Saved %s", psver_path)

        return written

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        loaded = self.n_ps is not None
        if loaded:
            return (
                f"<ISCEPSLoader(work_dir='{self.work_dir}', n_ps={self.n_ps}, "
                f"n_image={self.n_image})>"
            )
        return f"<ISCEPSLoader(work_dir='{self.work_dir}', not loaded)>"


# ============================================================================
# ISCESBLoader — SB (Small Baseline) mode loader
# ============================================================================

class ISCESBLoader(ISCEPSLoader):
    """Load ISCE-format SB (Small Baseline) data, mirroring ``sb_load_initial_isce.m``.

    This class extends ``ISCEPSLoader`` and overrides the baseline calculation
    logic to handle SB mode, where interferograms are formed between arbitrary
    image pairs (not just master-slave pairs).

    Key differences from PS mode:
    - Reads ``ifgday.1.in`` to get interferogram date pairs
    - Calculates interferogram baselines: ``bperp_ifg = bperp(image2) - bperp(image1)``
    - ``n_ifg`` > ``n_image`` (multiple interferograms per image)
    - Saves ``ifgday`` and ``ifgday_ix`` arrays

    MATLAB correspondence
    ---------------------
    This class replaces ``sb_load_initial_isce.m``. The main difference from
    PS mode is in baseline calculation (lines 85-92 of sb_load_initial_isce.m):
    ```matlab
    bperp = bperp(day_ix);  % sort by date
    bperp = [bperp(1:master_ix-1);0;bperp(master_ix:end)];  % insert master
    bperp = bperp(ifgday_ix(:,2)) - bperp(ifgday_ix(:,1));  % interferogram baselines
    ```
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        psver: int = 1,
    ) -> None:
        super().__init__(work_dir, psver)
        # SB-specific fields
        self.ifgday: Optional[np.ndarray] = None          # (n_ifg, 2) int32 date pairs
        self.ifgday_ix: Optional[np.ndarray] = None        # (n_ifg, 2) int32 image indices
        # Extend _FILE_NAMES to include ifgday
        self._FILE_NAMES = dict(self._FILE_NAMES)
        self._FILE_NAMES["ifgday"] = "ifgday.1.in"

    def load(self) -> "ISCESBLoader":
        """Execute the SB initial-load pipeline.

        This method extends the PS loader logic with SB-specific steps:
        1. Load interferogram date pairs from ``ifgday.1.in``
        2. Calculate interferogram baselines from image baselines
        3. Verify all interferogram dates exist in the image date list

        Returns *self* for method chaining.
        """
        logger.info("Loading ISCE SB data from: %s", self.work_dir)

        # ==============================================================
        # 1. Load dates (same as PS mode)
        # ==============================================================
        day_path = self._resolve("day")
        day_raw = np.loadtxt(str(day_path), dtype=np.int64)
        slave_ordinals = np.array(
            [self._yyyymmdd_to_ordinal(int(d)) for d in day_raw],
            dtype=np.int32,
        )
        self.day_ix = np.argsort(slave_ordinals).astype(np.int32)
        slave_ordinals = slave_ordinals[self.day_ix]

        master_path = self._resolve("master_day")
        master_yyyymmdd = int(np.loadtxt(str(master_path)))
        master_ord = self._yyyymmdd_to_ordinal(master_yyyymmdd)
        self.master_day = np.int32(master_ord)
        # Store master_yyyymmdd for later use (e.g., calamp parsing)
        self._master_yyyymmdd = master_yyyymmdd

        master_ix_0 = int(np.sum(slave_ordinals < master_ord))
        self.master_ix = np.int32(master_ix_0)

        self.day = np.insert(slave_ordinals, master_ix_0, master_ord).astype(
            np.int32
        )
        self.n_image = len(self.day)
        logger.info(
            "Dates loaded: %d slaves + 1 master → %d images, master_ix=%d",
            len(slave_ordinals),
            self.n_image,
            self.master_ix,
        )

        # ==============================================================
        # 2. Load interferogram dates (SB-specific)
        # ==============================================================
        # Resolve ifgday.1.in file (search in work_dir then parent)
        ifgday_name = "ifgday.1.in"
        ifgday_path = self.work_dir / ifgday_name
        if not ifgday_path.is_file():
            ifgday_path = self.work_dir.parent / ifgday_name
            if not ifgday_path.is_file():
                raise FileNotFoundError(
                    f"{ifgday_name} not found in {self.work_dir} or {self.work_dir.parent}"
                )
        ifgday_raw = np.loadtxt(str(ifgday_path), dtype=np.int64)
        if ifgday_raw.ndim == 1:
            # Handle single interferogram case
            ifgday_raw = ifgday_raw.reshape(1, -1)
        if ifgday_raw.shape[1] != 2:
            raise ValueError(
                f"ifgday.1.in must have 2 columns (got {ifgday_raw.shape[1]})"
            )

        # Convert to ordinals
        ifgday_ordinals = np.zeros((ifgday_raw.shape[0], 2), dtype=np.int32)
        for i in range(ifgday_raw.shape[0]):
            ifgday_ordinals[i, 0] = self._yyyymmdd_to_ordinal(int(ifgday_raw[i, 0]))
            ifgday_ordinals[i, 1] = self._yyyymmdd_to_ordinal(int(ifgday_raw[i, 1]))

        self.n_ifg = ifgday_ordinals.shape[0]
        logger.info("Interferograms loaded: n_ifg=%d", self.n_ifg)

        # Map interferogram dates to image indices
        # MATLAB: [found_true, ifgday_ix] = ismember(ifgday, day)
        ifgday_ix = np.zeros((self.n_ifg, 2), dtype=np.int32)
        for i in range(self.n_ifg):
            for j in range(2):
                matches = np.where(self.day == ifgday_ordinals[i, j])[0]
                if len(matches) == 0:
                    raise ValueError(
                        f"Interferogram date {ifgday_raw[i, j]} "
                        f"(ordinal {ifgday_ordinals[i, j]}) not found in day list"
                    )
                ifgday_ix[i, j] = matches[0]

        self.ifgday = ifgday_ordinals
        self.ifgday_ix = ifgday_ix
        logger.info(
            "ifgday_ix range: [%d, %d] → [%d, %d]",
            ifgday_ix[:, 0].min(),
            ifgday_ix[:, 0].max(),
            ifgday_ix[:, 1].min(),
            ifgday_ix[:, 1].max(),
        )

        # ==============================================================
        # 3. Calculate interferogram baselines (SB-specific)
        # ==============================================================
        bperp_path = self._resolve("bperp")
        bperp_raw = np.loadtxt(str(bperp_path), dtype=np.float64)
        bperp_sorted = bperp_raw[self.day_ix]
        # Insert 0 for master-master baseline
        bperp_image = np.insert(bperp_sorted, master_ix_0, 0.0)

        # Calculate interferogram baselines
        # MATLAB: bperp = bperp(ifgday_ix(:,2)) - bperp(ifgday_ix(:,1))
        self.bperp = bperp_image[ifgday_ix[:, 1]] - bperp_image[ifgday_ix[:, 0]]
        logger.info(
            "Baselines loaded: %d images → %d interferograms",
            self.n_image,
            self.n_ifg,
        )

        # ==============================================================
        # 4-12. Rest of the pipeline (same as PS mode)
        # ==============================================================
        # Heading & wavelength
        heading_path = self._resolve("heading")
        heading = float(np.loadtxt(str(heading_path)))
        if heading == 0.0:
            raise ValueError("heading.1.in is empty or zero")
        try:
            cfg = StampsConfig()
            if cfg._loaded:
                cfg.setparm("heading", heading, new_flag=1)
        except Exception:
            pass
        logger.info("Heading: %.6f deg", heading)

        lambda_path = self._resolve("lambda")
        wavelength = float(np.loadtxt(str(lambda_path)))
        try:
            cfg = StampsConfig()
            if cfg._loaded:
                cfg.setparm("lambda", wavelength, new_flag=1)
        except Exception:
            pass
        logger.info("Wavelength: %.4f m", wavelength)

        # Radar coordinates
        ij_path = self._resolve("ij")
        ij = np.loadtxt(str(ij_path), dtype=np.int32)
        n_ps_raw = ij.shape[0]
        self.n_ps = n_ps_raw
        logger.info("PS candidates: n_ps=%d", self.n_ps)

        # SB calamp.out contains paired SLC calibration rows for candidate
        # selection. The loaded wrapped IFG phase is normalized below instead.
        self.calconst = np.ones(self.n_ifg, dtype=np.float64)
        logger.info(
            "Calibration constants: %d values loaded", len(self.calconst)
        )

        # Complex phase (n_ifg columns in SB mode)
        ph_path = self._resolve("ph")
        actual_bytes = ph_path.stat().st_size
        actual_cols = actual_bytes // (n_ps_raw * 8)
        if actual_cols != self.n_ifg:
            logger.warning(
                "Phase file has %d columns, expected %d for SB mode. "
                "Loading actual columns.",
                actual_cols,
                self.n_ifg,
            )
            n_ifg_actual = actual_cols
        else:
            n_ifg_actual = self.n_ifg

        ph = self._read_complex_phase(ph_path, n_ps_raw, n_ifg_actual)

        # Drop PS where more than 1 phase is zero, matching sb_load_initial_isce.
        zero_ph = np.sum(ph == 0, axis=1)
        nonzero_ix = zero_ph <= 1
        if not np.all(nonzero_ix):
            dropped = int(np.size(nonzero_ix) - np.count_nonzero(nonzero_ix))
            logger.info("Dropping %d SB candidates with more than one zero phase", dropped)
        ph = ph[nonzero_ix]
        ij = ij[nonzero_ix]
        self.n_ps = int(ph.shape[0])

        # SB mode stores wrapped phase only; normalize non-zero complex samples.
        nonzero_phase = ph != 0
        ph[nonzero_phase] = ph[nonzero_phase] / np.abs(ph[nonzero_phase])

        self.ph = ph
        logger.info("Phase loaded: shape=%s, dtype=%s", self.ph.shape, self.ph.dtype)

        # Lon/Lat → local XY
        ll_path = self._resolve("ll")
        lonlat = self._read_lonlat(ll_path, n_ps_raw).astype(np.float64)
        lonlat = lonlat[nonzero_ix]
        self.ll0 = (lonlat.max(axis=0) + lonlat.min(axis=0)) / 2.0

        xy = self._llh2local(lonlat, self.ll0) * 1000.0

        # Heading-based rotation
        theta = np.deg2rad(180.0 - heading)
        if theta > np.pi:
            theta -= 2 * np.pi

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotm = np.array([[cos_t, sin_t], [-sin_t, cos_t]])

        xy_T = xy.T
        xy_new = rotm @ xy_T

        if (
            np.ptp(xy_new[0]) < np.ptp(xy_T[0])
            and np.ptp(xy_new[1]) < np.ptp(xy_T[1])
        ):
            xy_T = xy_new
            logger.info("Rotating coordinates by %.2f degrees", np.rad2deg(theta))

        xy = xy_T.T.astype(np.float32)

        # Sort by ascending y then x
        sort_ix = np.lexsort((xy[:, 0], xy[:, 1]))
        xy = xy[sort_ix]

        ids = np.arange(1, self.n_ps + 1, dtype=np.float32).reshape(-1, 1)
        xy_rounded = np.round(xy * 1000) / 1000.0
        xy_full = np.hstack([ids, xy_rounded]).astype(np.float32)

        self.sort_ix = sort_ix.astype(np.int32)

        # Reorder everything by sort_ix
        self.ph = self.ph[sort_ix]
        self.ij = ij[sort_ix]
        self.ij[:, 0] = np.arange(1, self.n_ps + 1)
        self.lonlat = lonlat[sort_ix]
        self.xy = xy_full

        # Dispersion (D_A)
        da_path = self._resolve_optional("da")
        if da_path is not None:
            try:
                D_A = np.loadtxt(str(da_path), dtype=np.float64)
                if D_A.shape[0] == self.n_ps:
                    self.D_A = D_A[sort_ix]
                else:
                    logger.warning(
                        "D_A has %d entries, n_ps=%d — skipping",
                        D_A.shape[0],
                        self.n_ps,
                    )
            except Exception as exc:
                logger.warning("Failed to load D_A: %s", exc)

        # Height
        hgt_path = self._resolve_optional("hgt")
        if hgt_path is not None:
            hgt = self._read_hgt(hgt_path)
            if hgt.shape[0] == self.n_ps:
                self.hgt = hgt[sort_ix]
            else:
                logger.warning(
                    "hgt has %d entries, n_ps=%d — skipping",
                    hgt.shape[0],
                    self.n_ps,
                )

        for key, attr_name in [("inc", "inc"), ("la_ps", "la")]:
            angle_path = self._resolve_optional(key)
            if angle_path is None:
                continue
            angle = self._read_hgt(angle_path)
            if angle.shape[0] == n_ps_raw:
                setattr(self, attr_name, angle[nonzero_ix][sort_ix])
            elif angle.shape[0] == self.n_ps:
                setattr(self, attr_name, angle[sort_ix])
            else:
                logger.warning(
                    "%s has %d entries, n_ps=%d - skipping",
                    angle_path.name,
                    angle.shape[0],
                    self.n_ps,
                )

        # Width & Length
        try:
            width_path = self._resolve("width")
            self.width = int(np.loadtxt(str(width_path)))
        except FileNotFoundError:
            self.width = None

        try:
            len_path = self._resolve("len")
            self.length = int(np.loadtxt(str(len_path)))
        except FileNotFoundError:
            self.length = None

        # Per-PS baselines (bperp_mat)
        # SB mode: bperp_mat is (n_ps, n_ifg) — one baseline per PS per interferogram
        # For now, replicate the interferogram baselines for all PS
        # (baseline grids would require additional files)
        self.bperp_mat = np.tile(self.bperp.astype(np.float32), (self.n_ps, 1))
        logger.info(
            "bperp_mat: shape=%s (interferogram baselines replicated)",
            self.bperp_mat.shape,
        )

        logger.info("ISCE SB data loading complete.")
        return self

    def __repr__(self) -> str:
        loaded = self.n_ps is not None
        if loaded:
            return (
                f"<ISCESBLoader(work_dir='{self.work_dir}', n_ps={self.n_ps}, "
                f"n_image={self.n_image}, n_ifg={self.n_ifg})>"
            )
        return f"<ISCESBLoader(work_dir='{self.work_dir}', not loaded)>"


# ============================================================================
# Standalone utility: readcpx (port of readcpx.m)
# ============================================================================

def readcpx(
    path: Union[str, Path],
    width: int,
    lines: int = -1,
    endian: str = "=",
    skip_bytes: int = 0,
    precision: str = "float32",
) -> np.ndarray:
    """Read complex-interleaved binary file, port of ``readcpx.m``.

    Parameters
    ----------
    path : str or Path
        Binary file path.
    width : int
        Number of complex samples per row.
    lines : int
        Number of rows to read (``-1`` = all).
    endian : str
        NumPy byte-order char (``'='`` native, ``'<'`` little, ``'>'`` big).
        MATLAB 'n' (native) ↔ NumPy '='.
    skip_bytes : int
        Bytes to skip at the start of the file.
    precision : str
        NumPy dtype string for each float component (default ``'float32'``).

    Returns
    -------
    ndarray, shape (lines, width), dtype complex64
        Complex image matrix.

    MATLAB correspondence
    ---------------------
    ```matlab
    fid = fopen(fname, 'r', endian);
    fseek(fid, skipbytes, -1);
    vname = fread(fid, [width*2, lines], [precision, '=>single']).';
    vname = complex(vname(:,1:2:end), vname(:,2:2:end));
    ```

    MATLAB ``fread(…, [width*2, lines])`` reads *column-first* into a
    ``(width*2, lines)`` matrix, then transposes with ``.'`` giving
    ``(lines, width*2)``.  The complex conversion takes odd/even columns.

    In Python, ``np.fromfile`` reads row-major, so we reshape to
    ``(lines, width, 2)`` and combine.
    """
    path = Path(path)
    dt = np.dtype(precision).newbyteorder(endian)
    file_size = path.stat().st_size - skip_bytes
    max_lines = file_size // (width * 2 * dt.itemsize)

    if lines < 0 or lines > max_lines:
        lines = max_lines

    with open(path, "rb") as fh:
        if skip_bytes:
            fh.seek(skip_bytes)
        raw = np.fromfile(fh, dtype=dt, count=lines * width * 2)

    raw = raw.reshape(lines, width, 2)
    return (raw[:, :, 0] + 1j * raw[:, :, 1]).astype(np.complex64)


# ============================================================================
# __main__ — demo: load ISCE test data
# ============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    script_dir = Path(__file__).resolve().parent
    test_data_dir = script_dir.parent / "test_data"
    patch_dir = test_data_dir / "PATCH_890"

    if not patch_dir.is_dir():
        print(f"ERROR: PATCH directory not found at {patch_dir}")
        sys.exit(1)

    print("=" * 65)
    print(f"ISCE PS Loader demo — work_dir: {patch_dir}")
    print("=" * 65)

    loader = ISCEPSLoader(work_dir=patch_dir)
    loader.load()

    print(f"\n--- Loaded summary ---")
    print(f"  n_ps       = {loader.n_ps}")
    print(f"  n_image    = {loader.n_image}")
    print(f"  n_ifg      = {loader.n_ifg}")
    print(f"  master_ix  = {loader.master_ix} (0-based)")
    print(f"  master_day = {loader.master_day} "
          f"({ISCEPSLoader._ordinal_to_datestr(int(loader.master_day))})")
    print(f"  day range  = "
          f"{ISCEPSLoader._ordinal_to_datestr(int(loader.day[0]))} .. "
          f"{ISCEPSLoader._ordinal_to_datestr(int(loader.day[-1]))}")
    print(f"  ph.shape   = {loader.ph.shape}  dtype={loader.ph.dtype}")
    print(f"  xy.shape   = {loader.xy.shape}")
    print(f"  lonlat range: lon=[{loader.lonlat[:,0].min():.4f}, "
          f"{loader.lonlat[:,0].max():.4f}]  "
          f"lat=[{loader.lonlat[:,1].min():.4f}, "
          f"{loader.lonlat[:,1].max():.4f}]")
    print(f"  ll0        = {loader.ll0}")
    print(f"  bperp[:5]  = {loader.bperp[:5]}")

    if loader.hgt is not None:
        print(f"  hgt range  = [{loader.hgt.min():.1f}, {loader.hgt.max():.1f}] m")
    if loader.width is not None:
        print(f"  width      = {loader.width}")
    if loader.length is not None:
        print(f"  length     = {loader.length}")

    # Save cached output
    print("\n--- Saving cached .npz files ---")
    written = loader.save(out_dir=patch_dir)
    for name, p in written.items():
        size_kb = p.stat().st_size / 1024
        print(f"  {name}: {p.name}  ({size_kb:.0f} KB)")

    # Verify a round-trip
    print("\n--- Round-trip verification ---")
    ps_data = np.load(str(written["ps"]))
    print(f"  ps1.npz keys: {list(ps_data.keys())}")
    print(f"  n_ps from file: {ps_data['n_ps']}")
    print(f"  master_ix from file: {ps_data['master_ix']}")

    ph_data = np.load(str(written["ph"]))
    print(f"  ph1.npz ph.shape: {ph_data['ph'].shape}, dtype={ph_data['ph'].dtype}")

    print("\nDone.")
