#!/usr/bin/env python3
"""
prep_isce.py - Python replacement scaffold for StaMPS mt_prep_isce.

Version 1 targets the current Python pipeline boundary: generate the same
root metadata and PATCH_*/pscands.1.* files that ISCEPSLoader consumes.
The implementation will progressively replace calamp, selpsc_patch,
psclonlat, pscdem, and pscphase with native Python code.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np


ROOT_METADATA_FILES = (
    "day.1.in",
    "master_day.1.in",
    "bperp.1.in",
    "heading.1.in",
    "lambda.1.in",
    "width.txt",
    "len.txt",
    "parms.mat",
)


@dataclass(frozen=True)
class PatchBounds:
    """Patch bounds in StaMPS convention: 1-based, inclusive."""

    name: str
    start_rg: int
    end_rg: int
    start_az: int
    end_az: int
    start_rg_noover: int
    end_rg_noover: int
    start_az_noover: int
    end_az_noover: int


@dataclass(frozen=True)
class CalampEntry:
    """One line from calamp.out."""

    path: Path
    calibration: float


def build_patch_bounds(
    width: int,
    length: int,
    range_patches: int = 1,
    azimuth_patches: int = 1,
    range_overlap: int = 50,
    azimuth_overlap: int = 50,
) -> List[PatchBounds]:
    """Mirror mt_prep_isce patch splitting."""
    if width <= 0 or length <= 0:
        raise ValueError("width and length must be positive")
    if range_patches <= 0 or azimuth_patches <= 0:
        raise ValueError("patch counts must be positive")

    width_p = width // range_patches
    length_p = length // azimuth_patches
    patches: List[PatchBounds] = []
    ip = 0

    for irg in range(1, range_patches + 1):
        for iaz in range(1, azimuth_patches + 1):
            ip += 1
            start_rg_noover = width_p * (irg - 1) + 1
            end_rg_noover = width_p * irg if irg < range_patches else width
            start_az_noover = length_p * (iaz - 1) + 1
            end_az_noover = length_p * iaz if iaz < azimuth_patches else length

            start_rg = max(1, start_rg_noover - range_overlap)
            end_rg = min(width, end_rg_noover + range_overlap)
            start_az = max(1, start_az_noover - azimuth_overlap)
            end_az = min(length, end_az_noover + azimuth_overlap)

            patches.append(
                PatchBounds(
                    name=f"PATCH_{ip}",
                    start_rg=start_rg,
                    end_rg=end_rg,
                    start_az=start_az,
                    end_az=end_az,
                    start_rg_noover=start_rg_noover,
                    end_rg_noover=end_rg_noover,
                    start_az_noover=start_az_noover,
                    end_az_noover=end_az_noover,
                )
            )
    return patches


def write_patch_layout(output_dir: Path, patches: Sequence[PatchBounds]) -> None:
    """Create PATCH_* directories, patch.in, patch_noover.in, and patch.list."""
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_names: List[str] = []

    for patch in patches:
        patch_dir = output_dir / patch.name
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_names.append(patch.name)

        (patch_dir / "patch.in").write_text(
            "\n".join(
                [
                    str(patch.start_rg),
                    str(patch.end_rg),
                    str(patch.start_az),
                    str(patch.end_az),
                    "",
                ]
            ),
            encoding="ascii",
        )
        (patch_dir / "patch_noover.in").write_text(
            "\n".join(
                [
                    str(patch.start_rg_noover),
                    str(patch.end_rg_noover),
                    str(patch.start_az_noover),
                    str(patch.end_az_noover),
                    "",
                ]
            ),
            encoding="ascii",
        )

    (output_dir / "patch.list").write_text(
        "\n".join(patch_names) + "\n",
        encoding="ascii",
    )


def read_scalar_int(path: Path) -> int:
    """Read a one-value text file as int."""
    return int(float(path.read_text(encoding="ascii").strip()))


def _isce_xml_property_int(xml_path: Path, name: str) -> Optional[int]:
    """Read an integer property from an ISCE XML image metadata file."""
    if not xml_path.is_file():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    for prop in root.findall(".//property"):
        if prop.attrib.get("name") != name:
            continue
        value = prop.findtext("value")
        if value is None:
            continue
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _isce_xml_property_float(xml_path: Path, name: str) -> Optional[float]:
    """Read a float property from an ISCE XML metadata file."""
    if not xml_path.is_file():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    target = name.lower()
    for prop in root.findall(".//property"):
        if prop.attrib.get("name", "").lower() != target:
            continue
        value = prop.findtext("value")
        if value is None:
            continue
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _isce_xml_property_text_ci(xml_path: Path, name: str) -> Optional[str]:
    """Read an ISCE XML property by name, case-insensitively."""
    if not xml_path.is_file():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    wanted = name.lower()
    for prop in root.iter("property"):
        if prop.attrib.get("name", "").lower() != wanted:
            continue
        value = prop.find("value")
        if value is not None and value.text is not None:
            return value.text.strip()
    return None


def _isce_xml_property_text(xml_path: Path, name: str) -> Optional[str]:
    """Read a text property from an ISCE XML metadata file."""
    if not xml_path.is_file():
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    target = name.lower()
    for prop in root.findall(".//property"):
        if prop.attrib.get("name", "").lower() == target:
            value = prop.findtext("value")
            return value.strip() if value else None
    return None


def _isce_xml_numpy_dtype(xml_path: Path) -> Optional[np.dtype]:
    """Map ISCE image XML data_type to a NumPy dtype."""
    data_type = _isce_xml_property_text(xml_path, "data_type")
    if data_type is None:
        return None
    normalized = data_type.strip().upper()
    if normalized in {"FLOAT", "REAL4"}:
        return np.dtype("<f4")
    if normalized in {"DOUBLE", "REAL8"}:
        return np.dtype("<f8")
    return None


def _discover_isce_stack_slcs(isce_dir: Path) -> List[Path]:
    """Find merged ISCE stack SLCs such as SLC/YYYYMMDD/YYYYMMDD.slc.full."""
    slc_root = isce_dir / "SLC"
    if not slc_root.is_dir():
        return []
    slcs: List[Path] = []
    for date_dir in sorted(slc_root.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.isdigit():
            continue
        preferred = date_dir / f"{date_dir.name}.slc.full"
        if preferred.is_file():
            slcs.append(preferred)
            continue
        fallback = date_dir / f"{date_dir.name}.slc"
        if fallback.is_file():
            slcs.append(fallback)
    return slcs


def _slc_date(path: Path) -> Optional[int]:
    """Extract YYYYMMDD date from a merged SLC path."""
    for part in [path.stem, path.name, path.parent.name]:
        match = re.search(r"(19|20)\d{6}", part)
        if match:
            return int(match.group(0))
    return None


def _stack_root(isce_dir: Path) -> Path:
    """Return the surrounding ISCE stack root for a merged directory."""
    return isce_dir.parent if isce_dir.name.lower() == "merged" else isce_dir


def infer_master_date(isce_dir: Path, slcs: Optional[Sequence[Path]] = None) -> Optional[int]:
    """Infer the master/reference date from ISCE stack baseline/config naming."""
    stack_root = _stack_root(isce_dir)
    baseline_root = stack_root / "baselines"
    if baseline_root.is_dir():
        masters: List[int] = []
        for child in baseline_root.iterdir():
            if not child.is_dir():
                continue
            match = re.match(r"((?:19|20)\d{6})_((?:19|20)\d{6})$", child.name)
            if match:
                masters.append(int(match.group(1)))
        if masters:
            return max(set(masters), key=masters.count)

    configs = stack_root / "configs"
    if configs.is_dir():
        merge_dates = []
        for path in configs.glob("config_merge_*"):
            match = re.search(r"config_merge_((?:19|20)\d{6})$", path.name)
            if match:
                merge_dates.append(int(match.group(1)))
        if slcs is not None:
            slc_dates = {_slc_date(path) for path in slcs}
            slc_dates.discard(None)
            unique = [date for date in merge_dates if date in slc_dates]
            if unique:
                return unique[0]

    return None


def _parse_bperp_average(path: Path) -> Optional[float]:
    """Read average perpendicular baseline from an ISCE baseline text file."""
    values: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = re.search(r"Bperp\s*\(average\)\s*:\s*([-+0-9.eE]+)", line)
        if match:
            values.append(float(match.group(1)))
    if not values:
        return None
    return float(np.mean(values))


def infer_slave_bperp(isce_dir: Path, master_date: int, slave_dates: Sequence[int]) -> List[float]:
    """Infer one scalar Bperp value per slave date from ISCE baseline reports."""
    stack_root = _stack_root(isce_dir)
    baseline_root = stack_root / "baselines"
    values: List[float] = []
    for slave_date in slave_dates:
        if slave_date == master_date:
            continue
        candidates = [
            baseline_root / f"{master_date}_{slave_date}" / f"{master_date}_{slave_date}.txt",
            baseline_root / f"{slave_date}_{master_date}" / f"{slave_date}_{master_date}.txt",
        ]
        value = None
        for path in candidates:
            if path.is_file():
                value = _parse_bperp_average(path)
                if value is not None:
                    break
        values.append(0.0 if value is None else value)
    return values


def infer_wavelength(isce_dir: Path) -> Optional[float]:
    """Infer radar wavelength from reference burst XML files."""
    stack_root = _stack_root(isce_dir)
    for xml_path in sorted((stack_root / "reference").glob("IW*.xml")):
        value = _isce_xml_property_float(xml_path, "radarwavelength")
        if value is not None:
            return value
    return None


def infer_heading(isce_dir: Path) -> Optional[float]:
    """Infer StaMPS heading in degrees from ISCE topo log messages."""
    stack_root = _stack_root(isce_dir)
    headings: List[float] = []
    for log_path in [stack_root / "stack_step.log", stack_root / "isce.log"]:
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"Default Peg heading set to:\s*([-+0-9.eE]+)", text):
            headings.append(float(match.group(1)) * 180.0 / np.pi)
    if headings:
        return float(np.mean(headings))
    return None


def infer_width_length(isce_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    """Infer raster dimensions from common ISCE stack metadata files."""
    candidates = [
        *sorted((isce_dir / "SLC").glob("[0-9]*/[0-9]*.slc.full.xml")),
        *sorted((isce_dir / "SLC").glob("[0-9]*/[0-9]*.slc.xml")),
        isce_dir / "geom_reference" / "lat.rdr.full.xml",
        isce_dir / "geom_reference" / "lat.rdr.xml",
    ]
    for xml_path in candidates:
        width = _isce_xml_property_int(xml_path, "width")
        length = _isce_xml_property_int(xml_path, "length")
        if width is not None and length is not None:
            return width, length
    return None, None


def write_inferred_root_metadata(
    input_dir: Path,
    output_dir: Path,
    width: int,
    length: int,
    slcs: Sequence[Path],
    heading: Optional[float] = None,
    wavelength: Optional[float] = None,
) -> None:
    """Create StaMPS root metadata files when they can be inferred."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dated_slcs = [(date, path) for path in slcs if (date := _slc_date(path)) is not None]
    dated_slcs.sort(key=lambda item: item[0])
    dates = [date for date, _ in dated_slcs]
    master_date = infer_master_date(input_dir, slcs)

    (output_dir / "width.txt").write_text(f"{width}\n", encoding="ascii")
    (output_dir / "len.txt").write_text(f"{length}\n", encoding="ascii")
    if master_date is not None:
        slave_dates = [date for date in dates if date != master_date]
        (output_dir / "master_day.1.in").write_text(f"{master_date}\n", encoding="ascii")
        (output_dir / "day.1.in").write_text(
            "".join(f"{date}\n" for date in slave_dates),
            encoding="ascii",
        )
        bperp = infer_slave_bperp(input_dir, master_date, slave_dates)
        (output_dir / "bperp.1.in").write_text(
            "".join(f"{value:.12g}\n" for value in bperp),
            encoding="ascii",
        )

    heading_value = heading if heading is not None else infer_heading(input_dir)
    if heading_value is not None:
        (output_dir / "heading.1.in").write_text(f"{heading_value:.12g}\n", encoding="ascii")

    wavelength_value = wavelength if wavelength is not None else infer_wavelength(input_dir)
    if wavelength_value is not None:
        (output_dir / "lambda.1.in").write_text(f"{wavelength_value:.12g}\n", encoding="ascii")


def discover_secondary_slcs(isce_dir: Path) -> List[Path]:
    """Find SLCs using the layout expected by mt_prep_isce.

    This is intentionally conservative for the scaffold. Metadata auto-detect
    will be expanded after the first candidate-extraction port is validated.
    """
    slcs: List[Path] = []
    # Older mt_prep_isce variants used master/master.slc and */slave.slc.
    master = isce_dir / "master" / "master.slc"
    if master.is_file():
        slcs.append(master)
    ref = isce_dir / "reference" / "reference.slc"
    if ref.is_file():
        slcs.append(ref)
    slcs.extend(sorted(isce_dir.glob("[0-9]*/slave.slc")))
    slcs.extend(sorted(isce_dir.glob("*/secondary.slc")))
    slcs.extend(_discover_isce_stack_slcs(isce_dir))
    return slcs


def write_calamp_input(output_dir: Path, slcs: Iterable[Path]) -> Path:
    """Write calamp.in with absolute SLC paths."""
    calamp_in = output_dir / "calamp.in"
    lines = [str(path.resolve()) for path in slcs]
    calamp_in.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return calamp_in


def copy_root_metadata(input_dir: Path, output_dir: Path) -> None:
    """Copy root metadata files used by ISCEPSLoader when present."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_METADATA_FILES:
        src = input_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)


def _complex_dtype(precision: str, byteswap: bool) -> Tuple[np.dtype, int]:
    """Return scalar dtype and scalar count per complex sample."""
    precision = precision.lower()
    if precision.startswith("s"):
        return (np.dtype(">i2" if byteswap else "<i2"), 2)
    if precision.startswith("f"):
        return (np.dtype(">f4" if byteswap else "<f4"), 2)
    raise ValueError("precision must be 'f' for complex float32 or 's' for complex int16")


def _read_mask_chunk(mask_fh, width: int, n_rows: int) -> np.ndarray:
    """Read mask bytes for n_rows. Missing mask bytes are treated as unmasked."""
    if mask_fh is None:
        return np.zeros((n_rows, width), dtype=bool)
    raw = np.fromfile(mask_fh, dtype=np.uint8, count=n_rows * width)
    if raw.size < n_rows * width:
        padded = np.zeros(n_rows * width, dtype=np.uint8)
        padded[: raw.size] = raw
        raw = padded
    return raw.reshape(n_rows, width) != 0


def compute_calibration_constant(
    slc_path: Path,
    width: int,
    precision: str = "f",
    byteswap: bool = False,
    mask_path: Optional[Path] = None,
    rows_per_chunk: int = 256,
) -> float:
    """Port of `calamp.c` for one SLC.

    The calibration constant is the mean amplitude over pixels with
    `abs(pixel) > 0.001` and mask value equal to zero.
    """
    scalar_dtype, scalars_per_complex = _complex_dtype(precision, byteswap)
    scalar_size = scalar_dtype.itemsize
    row_bytes = width * scalars_per_complex * scalar_size
    file_size = slc_path.stat().st_size
    if file_size % row_bytes != 0:
        raise ValueError(f"{slc_path} size is not divisible by width/precision row size")
    n_rows = file_size // row_bytes

    amp_sum = 0.0
    n_valid = 0

    with slc_path.open("rb") as slc_fh:
        mask_fh = mask_path.open("rb") if mask_path is not None and mask_path.is_file() else None
        try:
            rows_done = 0
            while rows_done < n_rows:
                n_chunk = min(rows_per_chunk, n_rows - rows_done)
                raw = np.fromfile(
                    slc_fh,
                    dtype=scalar_dtype,
                    count=n_chunk * width * scalars_per_complex,
                )
                if raw.size == 0:
                    break
                raw = raw.reshape(-1, width, scalars_per_complex)
                real = raw[:, :, 0].astype(np.float32, copy=False)
                imag = raw[:, :, 1].astype(np.float32, copy=False)
                amp = np.hypot(real, imag)
                mask = _read_mask_chunk(mask_fh, width, raw.shape[0])
                valid = (amp > 0.001) & ~mask
                if np.any(valid):
                    amp_sum += float(amp[valid].sum(dtype=np.float64))
                    n_valid += int(valid.sum())
                rows_done += raw.shape[0]
        finally:
            if mask_fh is not None:
                mask_fh.close()

    return amp_sum / n_valid if n_valid else 0.0


def run_calamp(
    calamp_input: Path,
    width: int,
    output_path: Path,
    precision: str = "f",
    byteswap: bool = False,
    mask_path: Optional[Path] = None,
) -> List[Tuple[Path, float]]:
    """Port of `calamp`: write `<slc_path> <calibration>` lines."""
    slcs = [
        Path(line.strip())
        for line in calamp_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results: List[Tuple[Path, float]] = []
    for index, slc in enumerate(slcs, start=1):
        print(f"[prep_isce] calamp {index}/{len(slcs)}: {slc}", flush=True)
        value = compute_calibration_constant(
            slc,
            width=width,
            precision=precision,
            byteswap=byteswap,
            mask_path=mask_path,
        )
        results.append((slc, value))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{path} {value:.6g}\n" for path, value in results),
        encoding="utf-8",
    )
    return results


def read_patch_bounds(path: Path) -> PatchBounds:
    """Read a StaMPS patch.in file."""
    vals = [int(float(x)) for x in path.read_text(encoding="ascii").split()]
    if len(vals) < 4:
        raise ValueError(f"{path} must contain four patch bounds")
    return PatchBounds(
        name=path.parent.name,
        start_rg=vals[0],
        end_rg=vals[1],
        start_az=vals[2],
        end_az=vals[3],
        start_rg_noover=vals[0],
        end_rg_noover=vals[1],
        start_az_noover=vals[2],
        end_az_noover=vals[3],
    )


def read_calamp_output(path: Path) -> List[CalampEntry]:
    """Read calamp.out lines as path/calibration pairs."""
    entries: List[CalampEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Invalid calamp line in {path}: {line!r}")
        entry_path = Path(parts[0])
        if not entry_path.is_absolute():
            entry_path = path.parent / entry_path
        entries.append(CalampEntry(entry_path, float(parts[1])))
    if not entries:
        raise ValueError(f"No SLC entries found in {path}")
    return entries


def _read_complex_window(
    fh,
    width: int,
    row_0: int,
    start_rg_0: int,
    count: int,
    scalar_dtype: np.dtype,
) -> np.ndarray:
    """Read one row/range complex window from an SLC."""
    scalar_size = scalar_dtype.itemsize
    offset = (row_0 * width + start_rg_0) * 2 * scalar_size
    fh.seek(offset)
    raw = np.fromfile(fh, dtype=scalar_dtype, count=count * 2)
    if raw.size < count * 2:
        padded = np.zeros(count * 2, dtype=scalar_dtype)
        padded[: raw.size] = raw
        raw = padded
    raw = raw.reshape(count, 2)
    return raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)


def _format_float_6g(value: float) -> str:
    """Match default C++ stream formatting closely enough for legacy files."""
    return f"{value:.6g}"


def select_ps_candidates(
    calamp_out: Path,
    patch_in: Path,
    width: int,
    ij_out: Path,
    da_out: Path,
    mean_amp_out: Path,
    precision: str = "f",
    byteswap: bool = False,
    mask_path: Optional[Path] = None,
    ij_int_out: Optional[Path] = None,
    ij0_out: Optional[Path] = None,
    da_thresh: Optional[float] = None,
    master_amp_path: Optional[Path] = None,
) -> int:
    """Port of `selpsc_patch` for PS candidate selection.

    Returns the number of selected candidates.
    """
    patch = read_patch_bounds(patch_in)
    entries = read_calamp_output(calamp_out)
    if da_thresh is None:
        raise ValueError("da_thresh must be provided for Python selpsc_patch")
    if any(entry.calibration == 0 for entry in entries):
        raise ValueError("calibration constants must be non-zero")

    scalar_dtype, _ = _complex_dtype(precision, byteswap)
    start_rg_0 = patch.start_rg - 1
    end_rg_0 = patch.end_rg - 1
    start_az_0 = patch.start_az - 1
    end_az_0 = patch.end_az - 1
    patch_width = end_rg_0 - start_rg_0 + 1
    patch_lines = end_az_0 - start_az_0 + 1
    pick_higher = da_thresh < 0
    da_thresh_sq = da_thresh * da_thresh

    ij_out.parent.mkdir(parents=True, exist_ok=True)
    da_out.parent.mkdir(parents=True, exist_ok=True)
    mean_amp_out.parent.mkdir(parents=True, exist_ok=True)
    if ij_int_out is None:
        ij_int_out = Path(str(ij_out) + ".int")
    if ij0_out is None:
        ij0_out = Path(str(ij_out) + "0")

    slc_handles = [entry.path.open("rb") for entry in entries]
    mask_fh = mask_path.open("rb") if mask_path is not None and mask_path.is_file() else None
    master_fh = master_amp_path.open("rb") if master_amp_path is not None and master_amp_path.is_file() else None
    pscid = 0

    try:
        with (
            ij_out.open("w", encoding="ascii") as ij_fh,
            da_out.open("w", encoding="ascii") as da_fh,
            mean_amp_out.open("wb") as mean_fh,
            ij_int_out.open("wb") as ij_int_fh,
            ij0_out.open("w", encoding="ascii") as ij0_fh,
        ):
            calib = np.array([entry.calibration for entry in entries], dtype=np.float32)
            for y0 in range(start_az_0, end_az_0 + 1):
                rows = np.vstack(
                    [
                        _read_complex_window(
                            fh,
                            width=width,
                            row_0=y0,
                            start_rg_0=start_rg_0,
                            count=patch_width,
                            scalar_dtype=scalar_dtype,
                        )
                        for fh in slc_handles
                    ]
                )
                amp_norm = np.abs(rows) / calib[:, np.newaxis]

                if master_fh is not None:
                    master_line = _read_complex_window(
                        master_fh,
                        width=width,
                        row_0=y0,
                        start_rg_0=start_rg_0,
                        count=patch_width,
                        scalar_dtype=scalar_dtype,
                    )
                    master_amp = np.abs(master_line)
                    master_amp[master_amp == 0] = 1.0
                    amp_norm = amp_norm / master_amp[np.newaxis, :]

                if mask_fh is not None:
                    mask_fh.seek(y0 * width + start_rg_0)
                    mask = np.fromfile(mask_fh, dtype=np.uint8, count=patch_width)
                    if mask.size < patch_width:
                        padded = np.zeros(patch_width, dtype=np.uint8)
                        padded[: mask.size] = mask
                        mask = padded
                    masked = mask != 0
                else:
                    masked = np.zeros(patch_width, dtype=bool)

                valid_amp = amp_norm > 0.00005
                sumamp = np.where(valid_amp, amp_norm, 0.0).sum(axis=0, dtype=np.float64)
                sumampsq = np.square(
                    np.where(valid_amp, amp_norm, 0.0),
                    dtype=np.float32,
                ).sum(axis=0, dtype=np.float64)
                np.asarray(sumamp, dtype="<f4").tofile(mean_fh)

                with np.errstate(divide="ignore", invalid="ignore"):
                    d_sq = len(entries) * sumampsq / (sumamp * sumamp) - 1.0
                amp_0 = np.any(~valid_amp, axis=0)
                selected = np.isfinite(d_sq) & (sumamp > 0) & ~masked
                if pick_higher:
                    selected &= d_sq >= da_thresh_sq
                else:
                    selected &= d_sq < da_thresh_sq

                x_selected = np.flatnonzero(selected & ~amp_0)
                if x_selected.size:
                    da_values = np.sqrt(d_sq[x_selected])
                    for x_rel, da_value in zip(x_selected.tolist(), da_values.tolist()):
                        x0 = start_rg_0 + x_rel
                        pscid += 1
                        ij_fh.write(f"{pscid} {y0} {x0}\n")
                        np.array([x0, y0], dtype=">i4").tofile(ij_int_fh)
                        da_fh.write(_format_float_6g(float(da_value)) + "\n")

                x_zero = np.flatnonzero(selected & amp_0)
                for x_rel in x_zero.tolist():
                    x0 = start_rg_0 + x_rel
                    ij0_fh.write(f"{pscid} {y0} {x0}\n")
    finally:
        for fh in slc_handles:
            fh.close()
        if mask_fh is not None:
            mask_fh.close()
        if master_fh is not None:
            master_fh.close()

    return pscid


def read_candidate_ij(path: Path) -> np.ndarray:
    """Read pscands.1.ij as columns [id, azimuth_0based, range_0based]."""
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 3), dtype=np.int64)
    arr = np.loadtxt(path, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 3:
        raise ValueError(f"{path} must have at least three columns")
    return arr[:, :3]


def _read_float_samples(
    raster_path: Path,
    width: int,
    yx: np.ndarray,
    dtype: np.dtype = np.dtype("<f4"),
) -> np.ndarray:
    """Read scalar raster samples at candidate az/range coordinates."""
    if yx.size == 0:
        return np.empty((0,), dtype=dtype)
    scalar_size = dtype.itemsize
    out = np.empty((yx.shape[0],), dtype=dtype)
    with raster_path.open("rb") as fh:
        for y0 in np.unique(yx[:, 0]):
            cand_ix = np.flatnonzero(yx[:, 0] == y0)
            x_values = yx[cand_ix, 1].astype(np.int64, copy=False)
            x_min = int(x_values.min())
            x_max = int(x_values.max())
            count = x_max - x_min + 1
            fh.seek((int(y0) * width + x_min) * scalar_size)
            raw = np.fromfile(fh, dtype=dtype, count=count)
            if raw.size < count:
                padded = np.zeros(count, dtype=dtype)
                padded[: raw.size] = raw
                raw = padded
            out[cand_ix] = raw[x_values - x_min]
    return out


def _infer_raster_dtype(raster_path: Path, default: str = "<f4") -> np.dtype:
    """Infer scalar raster dtype from the sidecar ISCE XML, falling back to default."""
    xml_path = Path(str(raster_path) + ".xml")
    if xml_path.is_file():
        dtype = _isce_xml_numpy_dtype(xml_path)
        if dtype is not None:
            return dtype
    return np.dtype(default)


def _infer_raster_bands_scheme(raster_path: Path) -> Tuple[int, str]:
    """Infer ISCE raster band count and interleaving scheme."""
    xml_path = Path(str(raster_path) + ".xml")
    bands = _isce_xml_property_int(xml_path, "number_bands") if xml_path.is_file() else None
    scheme = _isce_xml_property_text_ci(xml_path, "scheme") if xml_path.is_file() else None
    return int(bands or 1), str(scheme or "BIP").upper()


def _read_float_band_samples(
    raster_path: Path,
    width: int,
    yx: np.ndarray,
    dtype: np.dtype = np.dtype("<f4"),
    band: int = 1,
    number_bands: int = 1,
    scheme: str = "BIP",
) -> np.ndarray:
    """Read samples from a possibly multi-band ISCE raster.

    Coordinates in ``yx`` are ``[azimuth_0based, range_0based]``.  ``band`` is
    1-based, matching GDAL/ISCE metadata.  The ISCE ``incLocal.rdr.full`` files
    used by topsStack are typically two-band BIL rasters.
    """
    if yx.size == 0:
        return np.empty((0,), dtype=dtype)
    if band < 1 or band > number_bands:
        raise ValueError(f"band {band} out of range for {number_bands}-band raster: {raster_path}")
    band0 = band - 1
    scheme = scheme.upper()
    scalar_size = dtype.itemsize
    out = np.empty((yx.shape[0],), dtype=dtype)
    with raster_path.open("rb") as fh:
        for y0 in np.unique(yx[:, 0]):
            cand_ix = np.flatnonzero(yx[:, 0] == y0)
            x_values = yx[cand_ix, 1].astype(np.int64, copy=False)
            x_min = int(x_values.min())
            x_max = int(x_values.max())
            count = x_max - x_min + 1
            y0 = int(y0)
            if scheme == "BIL":
                sample_index = (y0 * number_bands + band0) * width + x_min
                fh.seek(sample_index * scalar_size)
                raw = np.fromfile(fh, dtype=dtype, count=count)
                if raw.size < count:
                    padded = np.zeros(count, dtype=dtype)
                    padded[: raw.size] = raw
                    raw = padded
                out[cand_ix] = raw[x_values - x_min]
            elif scheme == "BIP":
                sample_index = (y0 * width + x_min) * number_bands
                fh.seek(sample_index * scalar_size)
                raw = np.fromfile(fh, dtype=dtype, count=count * number_bands)
                if raw.size < count * number_bands:
                    padded = np.zeros(count * number_bands, dtype=dtype)
                    padded[: raw.size] = raw
                    raw = padded
                raw = raw.reshape(count, number_bands)
                out[cand_ix] = raw[x_values - x_min, band0]
            elif scheme == "BSQ":
                # This assumes full image length can be inferred from file size.
                total_samples = raster_path.stat().st_size // scalar_size
                length = total_samples // (width * number_bands)
                sample_index = (band0 * length + y0) * width + x_min
                fh.seek(sample_index * scalar_size)
                raw = np.fromfile(fh, dtype=dtype, count=count)
                if raw.size < count:
                    padded = np.zeros(count, dtype=dtype)
                    padded[: raw.size] = raw
                    raw = padded
                out[cand_ix] = raw[x_values - x_min]
            else:
                raise ValueError(f"Unsupported ISCE raster scheme {scheme!r} for {raster_path}")
    return out


def extract_lonlat(
    lon_path: Path,
    lat_path: Path,
    ij_path: Path,
    width: int,
    output_path: Path,
    dtype: Optional[str] = None,
) -> np.ndarray:
    """Port of `psclonlat`: write interleaved float32 lon/lat samples."""
    ij = read_candidate_ij(ij_path)
    yx = ij[:, [1, 2]]
    lon_dt = np.dtype(dtype) if dtype else _infer_raster_dtype(lon_path)
    lat_dt = np.dtype(dtype) if dtype else _infer_raster_dtype(lat_path)
    lon = _read_float_samples(lon_path, width, yx, lon_dt)
    lat = _read_float_samples(lat_path, width, yx, lat_dt)
    out = np.column_stack([lon, lat]).astype("<f4", copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(output_path)
    return out


def extract_dem(
    dem_path: Path,
    ij_path: Path,
    width: int,
    output_path: Path,
    dtype: Optional[str] = None,
) -> np.ndarray:
    """Port of `pscdem`: write height samples for selected candidates."""
    ij = read_candidate_ij(ij_path)
    yx = ij[:, [1, 2]]
    dt = np.dtype(dtype) if dtype else _infer_raster_dtype(dem_path)
    out = _read_float_samples(dem_path, width, yx, dt)
    out = out.astype("<f4", copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(output_path)
    return out


def extract_angle_raster(
    angle_path: Path,
    ij_path: Path,
    width: int,
    output_path: Path,
    band: int = 1,
    dtype: Optional[str] = None,
    input_unit: str = "degrees",
) -> np.ndarray:
    """Extract per-candidate angle samples and write radians as float32."""
    ij = read_candidate_ij(ij_path)
    yx = ij[:, [1, 2]]
    dt = np.dtype(dtype) if dtype else _infer_raster_dtype(angle_path)
    number_bands, scheme = _infer_raster_bands_scheme(angle_path)
    out = _read_float_band_samples(
        angle_path,
        width,
        yx,
        dt,
        band=band,
        number_bands=number_bands,
        scheme=scheme,
    ).astype(np.float32, copy=False)
    if input_unit.lower().startswith("deg"):
        valid = out != 0
        out[valid] = np.deg2rad(out[valid])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.astype("<f4", copy=False).tofile(output_path)
    return out


def extract_phase(
    interferograms: Sequence[Path],
    ij_path: Path,
    width: int,
    output_path: Path,
    precision: str = "f",
    byteswap: bool = False,
) -> np.ndarray:
    """Port of `pscphase`: write complex samples by IFG then candidate."""
    ij = read_candidate_ij(ij_path)
    yx = ij[:, [1, 2]]
    scalar_dtype, _ = _complex_dtype(precision, byteswap)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ph = np.empty((len(interferograms), yx.shape[0]), dtype=np.complex64)
    with output_path.open("wb") as out_fh:
        for i, ifg_path in enumerate(interferograms):
            with ifg_path.open("rb") as ifg_fh:
                for j, (y0, x0) in enumerate(yx):
                    sample = _read_complex_window(
                        ifg_fh,
                        width=width,
                        row_0=int(y0),
                        start_rg_0=int(x0),
                        count=1,
                        scalar_dtype=scalar_dtype,
                    )[0]
                    ph[i, j] = sample
                    np.array([sample.real, sample.imag], dtype="<f4").tofile(out_fh)
    return ph


def extract_phase_from_slcs(
    slcs: Sequence[Path],
    master_date: int,
    ij_path: Path,
    width: int,
    output_path: Path,
    precision: str = "f",
    byteswap: bool = False,
) -> np.ndarray:
    """Write PS wrapped phase as slave * conj(master) samples from co-registered SLCs."""
    dated_slcs = [(date, path) for path in slcs if (date := _slc_date(path)) is not None]
    dated_slcs.sort(key=lambda item: item[0])
    master_matches = [path for date, path in dated_slcs if date == master_date]
    if not master_matches:
        raise FileNotFoundError(f"Master SLC for {master_date} not found")
    master_path = master_matches[0]
    slave_slcs = [(date, path) for date, path in dated_slcs if date != master_date]

    ij = read_candidate_ij(ij_path)
    yx = ij[:, [1, 2]]
    scalar_dtype, _ = _complex_dtype(precision, byteswap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ph = np.empty((len(slave_slcs), yx.shape[0]), dtype=np.complex64)
    if yx.size == 0:
        ph.tofile(output_path)
        return ph

    # pscands.1.ij is emitted in row order. Grouping by row turns millions of
    # tiny random reads into contiguous row-window reads.
    row_groups: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for y0 in np.unique(yx[:, 0]):
        cand_ix = np.flatnonzero(yx[:, 0] == y0)
        x_values = yx[cand_ix, 1].astype(np.int64, copy=False)
        row_groups.append((int(y0), cand_ix, x_values))

    with master_path.open("rb") as master_fh:
        slave_handles = [path.open("rb") for _date, path in slave_slcs]
        try:
            for y0, cand_ix, x_values in row_groups:
                x_min = int(x_values.min())
                x_max = int(x_values.max())
                count = x_max - x_min + 1
                x_rel = x_values - x_min
                master_line = _read_complex_window(
                    master_fh,
                    width=width,
                    row_0=y0,
                    start_rg_0=x_min,
                    count=count,
                    scalar_dtype=scalar_dtype,
                )
                master_samples = master_line[x_rel]
                conj_master = np.conj(master_samples)
                for i, slave_fh in enumerate(slave_handles):
                    slave_line = _read_complex_window(
                        slave_fh,
                        width=width,
                        row_0=y0,
                        start_rg_0=x_min,
                        count=count,
                        scalar_dtype=scalar_dtype,
                    )
                    ph[i, cand_ix] = slave_line[x_rel] * conj_master
        finally:
            for fh in slave_handles:
                fh.close()
    ph.tofile(output_path)
    return ph


def read_path_list(path: Path, skip_first_width: bool = False) -> List[Path]:
    """Read helper files such as pscphase.in/psclonlat.in/pscdem.in."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if skip_first_width and lines:
        lines = lines[1:]
    paths: List[Path] = []
    for line in lines:
        p = Path(line)
        if not p.is_absolute():
            p = path.parent / p
        paths.append(p)
    return paths


def write_ps1_h5_outputs(output_dir: Path, patches: Sequence[PatchBounds]) -> None:
    """Run the existing Step-1 loader/save path for each prepared patch."""
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))

    from data_loader import ISCEPSLoader  # pylint: disable=import-outside-toplevel
    from stamps_main import (  # pylint: disable=import-outside-toplevel
        _load_no_ps_info,
        _save_step1_auxiliary_angles,
        _save_no_ps_info,
        _save_ps1_h5,
    )

    for patch in patches:
        patch_dir = output_dir / patch.name
        no_ps_path = patch_dir / "no_ps_info.h5"
        stamps_step_no_ps = _load_no_ps_info(no_ps_path)
        loader = ISCEPSLoader(work_dir=patch_dir, psver=1)
        loader.load()
        _save_ps1_h5(loader, patch_dir / "ps1.h5")
        _save_step1_auxiliary_angles(loader, patch_dir)
        stamps_step_no_ps[0] = 0 if loader.n_ps and loader.n_ps > 0 else 1
        _save_no_ps_info(no_ps_path, stamps_step_no_ps)


def run_preparation(args: argparse.Namespace) -> None:
    """Prepare layout and inputs; candidate extraction is implemented next."""
    isce_dir = Path(args.isce_dir).resolve()
    output_dir = Path(args.output).resolve()

    width = args.width
    length = args.length
    inferred_width, inferred_length = infer_width_length(isce_dir)
    if width is None:
        width = inferred_width if inferred_width is not None else read_scalar_int(isce_dir / "width.txt")
    if length is None:
        length = inferred_length if inferred_length is not None else read_scalar_int(isce_dir / "len.txt")

    patches = build_patch_bounds(
        width=width,
        length=length,
        range_patches=args.range_patches,
        azimuth_patches=args.azimuth_patches,
        range_overlap=args.range_overlap,
        azimuth_overlap=args.azimuth_overlap,
    )
    write_patch_layout(output_dir, patches)
    copy_root_metadata(isce_dir, output_dir)

    slcs = discover_secondary_slcs(isce_dir)
    write_inferred_root_metadata(
        isce_dir,
        output_dir,
        width=width,
        length=length,
        slcs=slcs,
        heading=args.heading,
        wavelength=args.wavelength,
    )
    calamp_in = Path(args.calamp_input).resolve() if args.calamp_input else write_calamp_input(output_dir, slcs)
    calamp_out = Path(args.calamp_output).resolve() if args.calamp_output else output_dir / "calamp.out"
    mask_path = Path(args.mask_file).resolve() if args.mask_file else None

    if args.run_calamp or args.calamp_only:
        run_calamp(
            calamp_in,
            width=width,
            output_path=calamp_out,
            precision=args.precision,
            byteswap=args.byteswap,
            mask_path=mask_path,
        )
    if args.calamp_only:
        return

    if args.run_select or args.select_only:
        if not calamp_out.is_file():
            raise FileNotFoundError(f"calamp output not found: {calamp_out}")
        for patch_index, patch in enumerate(patches, start=1):
            patch_dir = output_dir / patch.name
            print(
                f"[prep_isce] selecting {patch.name} ({patch_index}/{len(patches)})",
                flush=True,
            )
            select_ps_candidates(
                calamp_out=calamp_out,
                patch_in=patch_dir / "patch.in",
                width=width,
                ij_out=patch_dir / "pscands.1.ij",
                da_out=patch_dir / "pscands.1.da",
                mean_amp_out=patch_dir / "mean_amp.flt",
                precision=args.precision,
                byteswap=args.byteswap,
                mask_path=mask_path,
                da_thresh=args.da_thresh,
            )
    if args.run_extract or args.extract_only:
        ifg_paths: List[Path] = []
        if args.ifg_list:
            ifg_paths = read_path_list(Path(args.ifg_list).resolve(), skip_first_width=args.ifg_list_has_width)
        lon_path = Path(args.lon).resolve() if args.lon else isce_dir / "geom_reference" / "lon.rdr.full"
        lat_path = Path(args.lat).resolve() if args.lat else isce_dir / "geom_reference" / "lat.rdr.full"
        dem_path = Path(args.dem).resolve() if args.dem else isce_dir / "geom_reference" / "hgt.rdr.full"
        inc_path = Path(args.inc).resolve() if args.inc else isce_dir / "geom_reference" / "incLocal.rdr.full"
        la_path = Path(args.look_angle).resolve() if args.look_angle else inc_path
        master_date = args.master_date if args.master_date else infer_master_date(isce_dir, slcs)

        for patch_index, patch in enumerate(patches, start=1):
            patch_dir = output_dir / patch.name
            ij_path = patch_dir / "pscands.1.ij"
            print(
                f"[prep_isce] extracting {patch.name} ({patch_index}/{len(patches)})",
                flush=True,
            )
            if lon_path is not None and lat_path is not None and lon_path.is_file() and lat_path.is_file():
                extract_lonlat(lon_path, lat_path, ij_path, width, patch_dir / "pscands.1.ll")
            if dem_path is not None and dem_path.is_file():
                extract_dem(dem_path, ij_path, width, patch_dir / "pscands.1.hgt")
            if inc_path is not None and inc_path.is_file():
                extract_angle_raster(
                    inc_path,
                    ij_path,
                    width,
                    patch_dir / "pscands.1.inc",
                    band=args.inc_band,
                )
            if args.write_look_angle and la_path is not None and la_path.is_file():
                extract_angle_raster(
                    la_path,
                    ij_path,
                    width,
                    patch_dir / "pscands.1.la",
                    band=args.look_angle_band,
                )
            if ifg_paths:
                extract_phase(
                    ifg_paths,
                    ij_path,
                    width,
                    patch_dir / "pscands.1.ph",
                    precision=args.precision,
                    byteswap=args.byteswap,
                )
            elif args.phase_from_slcs:
                if master_date is None:
                    raise ValueError("Cannot infer master date for --phase-from-slcs")
                extract_phase_from_slcs(
                    slcs,
                    master_date,
                    ij_path,
                    width,
                    patch_dir / "pscands.1.ph",
                    precision=args.precision,
                    byteswap=args.byteswap,
                )
    if args.write_ps1:
        write_ps1_h5_outputs(output_dir, patches)

    if args.extract_only:
        return
    if args.select_only:
        return
    if args.run_extract or args.write_ps1:
        return

    raise NotImplementedError(
        "prep_isce currently creates patch layout and can run Python calamp/select. "
        "Next step is porting psclonlat/pscdem/pscphase extraction."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ISCE products for the Python StaMPS pipeline."
    )
    parser.add_argument("isce_dir", help="ISCE project directory")
    parser.add_argument("--output", required=True, help="Prepared output directory")
    parser.add_argument("--da-thresh", type=float, default=0.4)
    parser.add_argument("--range-patches", type=int, default=1)
    parser.add_argument("--azimuth-patches", type=int, default=1)
    parser.add_argument("--range-overlap", type=int, default=50)
    parser.add_argument("--azimuth-overlap", type=int, default=50)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--run-calamp", action="store_true", help="Compute calamp.out")
    parser.add_argument("--calamp-only", action="store_true", help="Only compute calamp and exit")
    parser.add_argument("--run-select", action="store_true", help="Run PS candidate selection")
    parser.add_argument("--select-only", action="store_true", help="Only run PS candidate selection and exit")
    parser.add_argument("--run-extract", action="store_true", help="Extract lon/lat, DEM, and phase samples")
    parser.add_argument("--extract-only", action="store_true", help="Only run extraction and exit")
    parser.add_argument("--write-ps1", action="store_true", help="Write PATCH_*/ps1.h5 via ISCEPSLoader")
    parser.add_argument("--lon", default=None, help="Longitude float32 raster")
    parser.add_argument("--lat", default=None, help="Latitude float32 raster")
    parser.add_argument("--dem", default=None, help="DEM float32 raster")
    parser.add_argument("--inc", default=None, help="Incidence-angle raster; defaults to geom_reference/incLocal.rdr.full")
    parser.add_argument("--inc-band", type=int, default=2, help="1-based band for --inc; ISCE incLocal band 2 is local incidence")
    parser.add_argument("--look-angle", default=None, help="Look-angle raster; defaults to --inc when --write-look-angle is set")
    parser.add_argument("--look-angle-band", type=int, default=1, help="1-based band for --look-angle; ISCE incLocal band 1 is sensor look angle")
    parser.add_argument("--write-look-angle", action="store_true", help="Also extract look-angle samples to pscands.1.la/la1.h5")
    parser.add_argument("--ifg-list", default=None, help="Text file listing complex interferograms")
    parser.add_argument("--ifg-list-has-width", action="store_true", help="First line of --ifg-list is width")
    parser.add_argument("--phase-from-slcs", action="store_true", help="Build PS phase samples from co-registered SLCs")
    parser.add_argument("--master-date", type=int, default=None, help="Master/reference date as YYYYMMDD")
    parser.add_argument("--heading", type=float, default=None, help="Override satellite heading in degrees")
    parser.add_argument("--wavelength", type=float, default=None, help="Override radar wavelength in meters")
    parser.add_argument("--calamp-input", default=None, help="Existing calamp.in file")
    parser.add_argument("--calamp-output", default=None, help="Output calamp.out path")
    parser.add_argument("--precision", choices=["f", "s"], default="f")
    parser.add_argument("--byteswap", action="store_true")
    parser.add_argument("--mask-file", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_preparation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
