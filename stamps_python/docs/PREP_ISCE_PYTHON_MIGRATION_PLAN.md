# Python ISCE Preprocessor Migration Plan

This document scopes the replacement for `stamps_matlab/bin/mt_prep_isce`.
The goal is to prepare ISCE products for the Python StaMPS pipeline without
requiring MATLAB-format preprocessing outputs or Linux-only shell/C utilities.

## Current MATLAB/Shell Flow

`mt_prep_isce` is a `csh` orchestration script. For PS mode it performs:

1. Extract ISCE metadata.
2. Build `calamp.in` from `reference/reference.slc` and `*/secondary.slc`.
3. Run `calamp` to compute amplitude calibration constants.
4. Create patch directories and write `patch.in`, `patch_noover.in`, and
   `patch.list`.
5. Write helper parameter files:
   - `selpsc.in`
   - `pscphase.in`
   - `psclonlat.in`
   - `pscdem.in`
6. Run `mt_extract_cands`, which calls:
   - `selpsc_patch` or `selsbc_patch`
   - `psclonlat`
   - `pscdem`
   - `pscphase`

The final files consumed by the existing Python `ISCEPSLoader` are:

- Root:
  - `parms.mat`
  - `patch.list`
  - `day.1.in`
  - `master_day.1.in`
  - `bperp.1.in`
  - `heading.1.in`
  - `lambda.1.in`
  - `width.txt`
  - `len.txt`
  - `calamp.out`
- Per patch:
  - `patch.in`
  - `patch_noover.in`
  - `pscands.1.ij`
  - `pscands.1.ij.int`
  - `pscands.1.ij0`
  - `pscands.1.da`
  - `pscands.1.ll`
  - `pscands.1.hgt`
  - `pscands.1.ph`
  - `mean_amp.flt`

## Windows Portability Assessment

The C/C++ utilities are mostly portable but not the full workflow.

Compiles with MinGW on the current Windows environment:

- `calamp`
- `cpxsum`
- `pscphase`
- `psclonlat`
- `pscdem`

Requires small source fixes:

- `selpsc_patch`
- `selsbc_patch`

Observed issues are ordinary C++ compatibility problems such as missing
`int32_t` declarations. These can be fixed with includes such as `<cstdint>`.

The larger blocker is not compilation. The shell workflow depends on `csh`,
`ls`, `cat`, `grep`, `awk/gawk`, relative Unix paths, `$STAMPS`, and helper
scripts. Also, the checked-in `mt_prep_isce` calls `mt_extract_info_isce`, but
that script is not present in this repository. Only the more generic
`mt_extract_info` exists. Therefore the original preprocessing chain is not a
complete, self-contained Windows target.

## Recommendation

Do not port `mt_prep_isce` literally. Replace it with a native Python
preprocessor:

```text
stamps_python/prep_isce.py
```

The first version should preserve the current Python Step-1 boundary by
writing the same `pscands.1.*` and root metadata files that `ISCEPSLoader`
already reads. After that is validated, a second version can write `ps1.h5`
directly and let the pipeline start at Step 2.

## Version 1 Scope: Python-Compatible Legacy Inputs

Version 1 should generate the current `ISCEPSLoader` inputs, not MATLAB `.mat`
outputs.

Command shape:

```powershell
python stamps_python/prep_isce.py <isce_project_dir> `
  --output <prepared_project_dir> `
  --da-thresh 0.4 `
  --range-patches 1 `
  --azimuth-patches 1 `
  --range-overlap 50 `
  --azimuth-overlap 50
```

Expected root outputs:

- `patch.list`
- `day.1.in`
- `master_day.1.in`
- `bperp.1.in`
- `heading.1.in`
- `lambda.1.in`
- `width.txt`
- `len.txt`
- `calamp.out`
- optional `parms.mat` copied from a template or generated later by Python

Expected patch outputs:

- `PATCH_*/patch.in`
- `PATCH_*/patch_noover.in`
- `PATCH_*/pscands.1.ij`
- `PATCH_*/pscands.1.ij.int`
- `PATCH_*/pscands.1.ij0`
- `PATCH_*/pscands.1.da`
- `PATCH_*/pscands.1.ll`
- `PATCH_*/pscands.1.hgt`
- `PATCH_*/pscands.1.ph`
- `PATCH_*/mean_amp.flt`

No `.mat` processing output should be written by the preprocessor.

## Core Algorithm Mapping

### Patch Layout

Follow `mt_prep_isce`:

- `width_p = width / range_patches`
- `length_p = length / azimuth_patches`
- patch ranges are 1-based inclusive in `patch.in`
- overlap is applied to `patch.in`
- no-overlap bounds are written to `patch_noover.in`

`pscands.1.ij` stores:

```text
candidate_id azimuth_0based range_0based
```

This matches the current Python loader expectation.

### Amplitude Calibration

Port `calamp.c`:

- Input: list of complex SLCs and raster `width`.
- For each SLC, scan all pixels.
- Amplitude is `abs(complex_pixel)`.
- Ignore amplitudes `<= 0.001` and masked pixels.
- Calibration constant is mean non-zero amplitude.
- Output line: `<slc_path> <calibration_constant>`.

Support first:

- complex float32 SLCs
- little-endian native data
- optional mask

Support later if needed:

- complex int16 input
- byte-swapped input
- Sun raster headers

### PS Candidate Selection

Port `selpsc_patch.c` for PS mode:

For each pixel in a patch:

1. Read that pixel from every calibrated SLC.
2. Compute normalized amplitude:

   ```text
   amp_i = abs(slc_i_pixel) / calib_i / abs(master_amp_pixel)
   ```

   If no separate master amplitude file is used, `master_amp_pixel = 1`.

3. If any normalized amplitude is `<= 0.00005`, mark the pixel as zero-amplitude
   and do not select it.
4. Compute:

   ```text
   D_sq = n_files * sum(amp_i^2) / sum(amp_i)^2 - 1
   D_A = sqrt(D_sq)
   ```

5. Select when:

   ```text
   D_A < da_thresh
   ```

   If `da_thresh` is negative, select the complementary high-dispersion set,
   matching the original `pick_higher` branch.

6. Write:
   - `pscands.1.ij`
   - `pscands.1.ij.int`
   - `pscands.1.ij0`
   - `pscands.1.da`
   - `mean_amp.flt`

### Lon/Lat Extraction

Port `psclonlat.c`:

- Input rasters: `lon.raw`, `lat.raw`, float32, radar grid.
- For each selected candidate `(azimuth, range)`, read:

  ```text
  offset = (azimuth * width + range) * 4
  ```

- Write interleaved float32 `[lon, lat]` to `pscands.1.ll`.

### DEM Extraction

Port `pscdem.c`:

- Input raster: `dem.raw`, float32 initially.
- Same candidate offsets as lon/lat.
- Write float32 height to `pscands.1.hgt`.

Double precision DEM can be added behind an option if an input dataset needs it.

### Phase Extraction

Port `pscphase.c`:

- Input interferograms: `*/isce_minrefdem.int`, complex float32.
- For each interferogram and selected candidate, read:

  ```text
  offset = (azimuth * width + range) * 8
  ```

- Write complex float32 samples in the same order used by the current loader:
  all candidates for IFG 1, then all candidates for IFG 2, etc.

## ISCE Metadata Extraction

The missing or incomplete part is metadata extraction, formerly
`mt_extract_info_isce`.

Version 1 should support two input modes:

1. **Explicit metadata mode**:
   The user supplies or already has root files:
   - `day.1.in`
   - `master_day.1.in`
   - `bperp.1.in`
   - `heading.1.in`
   - `lambda.1.in`
   - `width.txt`
   - `len.txt`

   This is enough to replace candidate extraction and patch preparation first.

2. **Auto-detect mode**:
   Parse common ISCE file names and XML/VRT metadata to infer:
   - raster width/length
   - master date
   - secondary dates
   - wavelength
   - heading
   - scalar or grid baselines

Auto-detect mode should be implemented after Version 1 is validated because
ISCE layouts vary between workflows.

## Proposed Implementation Steps

1. Add `stamps_python/prep_isce.py` with CLI parsing and patch layout creation.
   Done. The scaffold writes `PATCH_*`, `patch.in`, `patch_noover.in`,
   `patch.list`, and `calamp.in`.
2. Implement raster readers:
   - complex float32 by row/range window
   - float32 scalar raster by candidate indices
   - optional mask reader
3. Implement `calamp` in Python and validate `calamp.out` against
   `test_data/calamp.out`.
   Partially done. `prep_isce.py` now contains a Python `calamp` port with
   complex float32, complex int16, byte-swap, and mask support. The original
   fixture does not include source `.slc` files, so `test_data/calamp.out`
   cannot be regenerated directly. Synthetic float32+mask data match the
   original C `calamp` output exactly after its default 6-significant-digit
   formatting. The original C `calamp` short/byte-swap path crashes under the
   current Windows/MinGW environment, so that branch was validated against a
   direct NumPy expected-value calculation instead.
4. Implement PS candidate selection and validate:
   - `pscands.1.ij`
   - `pscands.1.da`
   - `mean_amp.flt`
   against `test_data/PATCH_437` and `PATCH_438`.
   Partially done. `prep_isce.py` now contains a PS-mode `selpsc_patch` port
   and CLI wiring through `--run-select` / `--select-only`. Because the
   original C `selpsc_patch` is not reliable on Windows even after adding the
   missing `int32_t` header, the current validation uses an independent NumPy
   expected-value calculation on synthetic complex-float SLCs. Candidate text
   indices and big-endian `pscands.1.ij.int` matched exactly. `mean_amp.flt`
   matched to float32 rounding (`~4.8e-7`), and `pscands.1.da` matched to
   better than `1e-6`.
5. Implement lon/lat, DEM, and phase extraction and validate:
   - `pscands.1.ll`
   - `pscands.1.hgt`
   - `pscands.1.ph`
   Done for the Python helper layer. `prep_isce.py` now includes ports of
   `psclonlat`, `pscdem`, and `pscphase` and CLI wiring via
   `--run-extract` / `--extract-only`. Synthetic raster validation confirms
   lon/lat and DEM samples match expected candidate offsets exactly, and phase
   samples are written in the same IFG-major order expected by
   `ISCEPSLoader`.
6. Run existing `StampsRunner --start 1 --end 8` on the preprocessor output.
   A synthetic end-to-end preprocessor fixture now passes the Step-1 loader
   boundary. `prep_isce.py` generated patch layout, `calamp.out`,
   `pscands.1.ij`, `pscands.1.da`, `mean_amp.flt`, `pscands.1.ll`,
   `pscands.1.hgt`, and `pscands.1.ph` from small ISCE-like rasters. After
   adding minimal root metadata files, `ISCEPSLoader` loaded the prepared
   `PATCH_1` successfully:
   - `n_ps`: `80`
   - `n_ifg`: `4`
   - `ph`: `(80, 4)`
   - `lonlat`: `(80, 2)`
   - `hgt`: `(80,)`
   - `bperp_mat`: `(80, 3)`
   - `calconst`: `3`
   The validation report is stored at
   `validation_runs/prep_isce_e2e_synthetic/report.json`.
7. Add a regression command that checks candidate counts and Step-1 shapes.
   Done. Run:
   ```powershell
   python stamps_python/tests/prep_isce_synthetic_check.py
   ```
   The script creates a synthetic ISCE-like fixture, runs `prep_isce.py`, loads
   the generated `PATCH_1` with `ISCEPSLoader`, and writes
   `validation_runs/prep_isce_synthetic_check/report.json`. The default run
   removes bulky generated fixture files after a pass; use `--keep` to retain
   them for debugging.

## Validation Targets

Use the existing two-patch dataset as the first acceptance fixture.

Expected candidate counts from the current fixture:

- `PATCH_437`: `50468`
- `PATCH_438`: `65193`

Initial validation tolerances:

- `pscands.1.ij`: exact match.
- `pscands.1.da`: float tolerance around `1e-6`.
- `pscands.1.ll`: exact bytes or float tolerance around `1e-7`.
- `pscands.1.hgt`: exact bytes or float tolerance around `1e-6`.
- `pscands.1.ph`: exact bytes if source rasters match and dtype/endian handling is
  identical.
- Step 1 `ps1.h5`: exact metadata and shape match; phase arrays should match
  existing prepared data.

## Version 2 Scope: Direct `ps1.h5`

After Version 1 is validated, add an option:

```powershell
python stamps_python/prep_isce.py <isce_project_dir> --output <prepared_project_dir> --write-ps1
```

First-stage `--write-ps1` is now implemented. It still writes the
legacy-compatible `pscands.1.*` files, then reuses the validated
`ISCEPSLoader` and `_save_ps1_h5()` path to write `PATCH_*/ps1.h5`. This
brings Step 1 into the preprocessor entry point without duplicating Step-1
metadata, sorting, coordinate rotation, or HDF5 layout logic.

The synthetic regression now checks this path:

```powershell
python stamps_python/tests/prep_isce_synthetic_check.py
```

Current result:

- `ps1_exists`: `true`
- `ps1_ph_shape`: `[80, 4]`
- `ps1_n_ps`: `80`
- `runner_step1_reused_ps1`: `true`

`StampsRunner.run_step_1()` now treats an existing `PATCH_*/ps1.h5` as a valid
Step-1 product. It updates `no_ps_info.h5` and returns without re-reading raw
candidate files. This allows the preprocessor output to enter the main pipeline
without Step 1 overwriting the prepared HDF5.

The next refinement can bypass `pscands.1.*` internally and write Python-native
Step-1 HDF5 directly. At that point `StampsRunner` can either:

- start at Step 2, or
- run from Step 1 and reuse `ps1.h5`.

Even after direct HDF5 writing is added, the preprocessor should keep an option
to emit legacy-compatible files for debugging and cross-checking.

## Open Questions

- Which ISCE directory layouts must be supported first: the current fixture
  layout only, or multiple ISCE2/ISCE3 variants?
- Should `parms.mat` remain the configuration source initially, or should this
  preprocessor introduce a Python-native config file and generate `parms.mat`
  only for compatibility?
- Is small-baseline preprocessing required in the first pass, or can Version 1
  target PS mode only?
- Do we need byte-swapped and complex int16 input support for real project data,
  or can the first pass require complex float32?

## Practical Conclusion

Porting the C utilities to Windows is possible, but it would keep the weakest
parts of the old workflow: shell orchestration, platform-specific tools, and
legacy StaMPS intermediate files. A Python preprocessor is the cleaner path and
fits the already validated Python Steps 1-8 pipeline.
