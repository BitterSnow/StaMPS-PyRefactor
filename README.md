# StaMPS-PyRefactor

StaMPS-PyRefactor is a Python port and workflow refactor of the MATLAB
StaMPS Persistent Scatterer InSAR processing chain. The repository provides the
translated Python implementation under `stamps_python/`. The original MATLAB
StaMPS source is used as a local translation reference, but is not vendored in
this Git repository.

The current focus is PS-InSAR processing for ISCE/ISCE2 stack outputs. The goal
is not to reproduce the MATLAB plotting GUI, but to make the processing and
export pipeline runnable from Python on Windows/Linux with HDF5-based
intermediate products.

This is an independent refactor/translation project and is not the official
StaMPS distribution.

## License

StaMPS is licensed under the GNU General Public License v3.0. This Python
refactor is derived from the StaMPS workflow and translation reference, so the
project is released under the same license: GPL-3.0.

See [LICENSE](LICENSE).

## Repository Layout

```text
stamps_python/      Python implementation of preprocessing, Steps 1-8, and export
OPERATION_MANUAL.md Historical operation notes
```

Large datasets, generated HDF5/MAT files, GeoPackages, Shapefiles, logs, and
validation runs are intentionally ignored by Git.

## Main Python Entry Points

| Script | Purpose |
| --- | --- |
| `stamps_python/prep_isce.py` | Read ISCE/ISCE2 stack products and prepare Python-compatible StaMPS patch data |
| `stamps_python/stamps_main.py` | Run StaMPS Steps 1-8 |
| `stamps_python/export_results.py` | Export velocity and displacement time series to GeoPackage or Shapefile |
| `stamps_python/validate_outputs.py` | Helper checks for generated outputs |

## Completed Work

- ISCE/ISCE2 PS preprocessing has been migrated from `mt_prep_isce` concepts to
  Python-oriented outputs.
- Patch preparation can produce Python-compatible HDF5 inputs and no longer
  depends on MATLAB `.mat` products for the main flow.
- ISCE2 `merged` directories can now be used directly as preprocessing input.
  The Python preprocessor can generate `day.1.in`, `master_day.1.in`,
  `bperp.1.in`, `heading.1.in`, `lambda.1.in`, `parms.json`, and
  `localparms.json` without creating or reading a MATLAB/StaMPS
  `INSAR_YYYYMMDD` directory.
- StaMPS PS workflow Steps 1-8 are implemented in Python:
  - Step 1: initial data loading
  - Step 2: gamma estimation
  - Step 3: PS selection, including gamma re-estimation
  - Step 4: weeding
  - Step 5: phase correction and patch merge
  - Step 6: unwrapping with `snaphu`
  - Step 7: SCLA estimation and smoothing
  - Step 8: SCN filtering
- Full-chain PS processing has been run on a real ISCE2 merged stack dataset.
- Step 3 gamma re-estimation performance has been improved:
  - unique grid-cell reuse
  - faster spectral smoothing
  - large HDF5 `bperp_mat` reads avoid slow fancy indexing
- Step 4/5 save and merge bottlenecks have been optimized.
- Incidence-angle extraction from ISCE2 `incLocal.rdr.full` is supported.
- ISCE2 small-baseline preprocessing can consume existing ISCE2 interferograms
  with `--small-baseline`, generating `ifgday.1.in`, `small_baselines.list`,
  paired-SLC candidate selection inputs, and SB-compatible `ps1.h5` products.
- Result export is implemented:
  - velocity in mm/yr
  - displacement time series in mm
  - `ps_deramp` support before export
  - GeoPackage output by default
  - Shapefile output retained as an option

## Current Validation Status

The PS-InSAR workflow has been run through Step 8 on a real ISCE2 SLC stack.
The validated merged outputs include:

- `ps2.h5`
- `ph2.h5`
- `pm2.h5`
- `phuw2.h5`
- `scla2.h5`
- `scla_smooth2.h5`
- `scn2.h5`

The exported GeoPackage test completed with:

- 1,651,830 point features
- `vel` plus 45 date fields
- default correction `v-dso`: SCLA + SCN correction followed by deramp

Validation datasets and generated outputs are not committed to Git because of
size and because each user will have their own ISCE/StaMPS project layout.

The preprocessing bootstrap path has also been validated with a clean input
view containing only:

- `SLC/`
- `geom_reference/`
- `baselines/`
- `input_file_YYYYMMDD`

That validation intentionally excluded `INSAR_YYYYMMDD`, `parms.mat`, and
`localparms.mat`. The generated Python metadata matched the native ISCE/StaMPS
reference values for heading and perpendicular baseline.

## Basic Usage

Install dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

Inspect preprocessing options:

```bash
python stamps_python/prep_isce.py --help
```

Prepare an ISCE2 SLC stack for the PS-InSAR workflow. The input directory should
be the ISCE2 `merged` directory produced by the SLC stack workflow, for example:

```text
merged/
  baselines/
  geom_reference/
    hgt.rdr.full
    incLocal.rdr.full
    lat.rdr.full
    lon.rdr.full
  SLC/
    20200101/
      20200101.slc.full
      20200101.slc.full.xml
    20200113/
      20200113.slc.full
      20200113.slc.full.xml
```

Run preprocessing from the repository root:

```bash
python stamps_python/prep_isce.py path/to/merged \
  --output validation_runs/prep_isce_ps \
  --reference-date 20200101 \
  --bootstrap-metadata \
  --range-patches 8 \
  --azimuth-patches 2 \
  --da-thresh 0.4 \
  --run-calamp \
  --run-select \
  --run-extract \
  --phase-from-slcs \
  --write-ps1
```

This command discovers co-registered SLCs from `SLC/YYYYMMDD/*.slc.full`,
infers `width.txt` and `len.txt`, reads the acquisition dates, computes
perpendicular baselines from `baselines/YYYYMMDD/YYYYMMDD`, computes heading
from `geom_reference/los.rdr.full`, reads wavelength from `input_file_*`, and
extracts longitude/latitude/DEM/incidence angle from `geom_reference/`.
`--bootstrap-metadata` writes Python-native `parms.json` and `localparms.json`
alongside the StaMPS-style text metadata files. `--reference-date` should be
passed explicitly for reproducible PS processing; `--master-date` remains as a
backward-compatible alias. `--da-thresh` controls the amplitude-dispersion
threshold used during PS candidate selection; the default is `0.4`.

To only test metadata generation without running candidate selection or phase
extraction:

```bash
python stamps_python/prep_isce.py path/to/merged \
  --output validation_runs/prep_isce_metadata_probe \
  --reference-date 20200101 \
  --bootstrap-metadata \
  --metadata-only \
  --range-patches 8 \
  --azimuth-patches 2
```

Example command matching the project full-stack validation settings, using
relative paths:

```bash
python stamps_python/prep_isce.py data/isce2_stack/merged \
  --output runs/prep_isce_ps_8x2 \
  --reference-date 20250911 \
  --bootstrap-metadata \
  --range-patches 8 \
  --azimuth-patches 2 \
  --da-thresh 0.4 \
  --run-calamp \
  --run-select \
  --run-extract \
  --phase-from-slcs \
  --write-ps1
```

Prepare existing ISCE2 small-baseline interferograms:

```bash
python stamps_python/prep_isce.py path/to/merged \
  --output path/to/prepared_sb \
  --small-baseline \
  --run-calamp \
  --run-select \
  --run-extract \
  --write-ps1
```

Run StaMPS Steps 1-8:

```bash
python stamps_python/stamps_main.py \
  --start 1 \
  --end 8 \
  --config path/to/prepared_project
```

Export deramped velocity and displacement time series to GeoPackage:

```bash
python stamps_python/export_results.py \
  --input-path path/to/prepared_project \
  --output-dir path/to/prepared_project/export \
  --format gpkg \
  --correction v-dso
```

Optional Shapefile export:

```bash
python stamps_python/export_results.py \
  --input-path path/to/prepared_project \
  --output-dir path/to/prepared_project/export \
  --format shp \
  --correction v-dso
```

## Important Notes

- The Python workflow targets PS-InSAR and can prepare ISCE2 small-baseline
  interferograms for the translated SB path. Full-chain SB validation should be
  repeated for each new stack geometry before production use.
- The project is designed around HDF5 outputs (`*.h5`) and JSON parameter files
  (`parms.json`, `localparms.json`) rather than MATLAB `.mat` files for the main
  Python path.
- For the direct-`merged` PS workflow, the preprocessor does not require the
  original `make_single_reference_stack_isce` output directory
  (`INSAR_YYYYMMDD`). It uses `SLC/`, `geom_reference/`, `baselines/`, and
  `input_file_*` directly.
- `snaphu` must be installed and available on `PATH`, or configured through the
  StaMPS parameters, for Step 6 unwrapping.
- Full processing can be CPU-, memory-, and disk-intensive. Generated products
  can be hundreds of MB to multiple GB.
- Shapefile has format limitations and is not ideal for dense time-series
  exports. GeoPackage is the recommended vector format.
- Some documentation files were created during iterative validation and may lag
  behind the code. Treat `stamps_python/` as the implementation source of truth.

## Known Gaps / Not Yet Complete

- Small Baseline workflow is not fully validated.
- MATLAB plotting functions are not ported as GUI/figure tools. Only the
  non-plotting velocity/time-series export path has been extracted.
- TRAIN/tropospheric correction integrations are not ported end to end.
- Cross-platform validation has focused mainly on Windows with Python 3.11.
- Numerical equivalence to MATLAB is validated progressively but not guaranteed
  for every branch and optional correction mode.
- Packaging is not yet formalized as an installable Python package.
- Automated tests are still limited compared with the size of the workflow.

## Citation / Attribution

This project is based on StaMPS (Stanford Method for Persistent Scatterers) and
uses the original MATLAB implementation as a translation reference. Please cite
the appropriate StaMPS publications and respect the GPL-3.0 license when using
or distributing this work.
