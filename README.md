# StaMPS-PyRefactor

StaMPS-PyRefactor is a Python port and workflow refactor of the MATLAB
StaMPS Persistent Scatterer InSAR processing chain. The repository keeps the
original MATLAB StaMPS source under `stamps_matlab/` for reference and provides
the translated Python implementation under `stamps_python/`.

The current focus is PS-InSAR processing for ISCE/ISCE2 stack outputs. The goal
is not to reproduce the MATLAB plotting GUI, but to make the processing and
export pipeline runnable from Python on Windows/Linux with HDF5-based
intermediate products.

This is an independent refactor/translation project and is not the official
StaMPS distribution.

## License

The original StaMPS code in this repository is licensed under the GNU General
Public License v3.0. This Python refactor is derived from and distributed with
that code, so the project is released under the same license: GPL-3.0.

See [LICENSE](LICENSE).

## Repository Layout

```text
stamps_matlab/      Original MATLAB StaMPS source used as translation reference
stamps_python/      Python implementation of preprocessing, Steps 1-8, and export
stamps_python/docs/ Migration plans, validation notes, and runbooks
OPERATION_MANUAL.md Historical operation notes
VALIDATION_PLAN.md  Translation validation checklist
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
- Result export is implemented:
  - velocity in mm/yr
  - displacement time series in mm
  - `ps_deramp` support before export
  - GeoPackage output by default
  - Shapefile output retained as an option

## Current Validation Status

The latest full-chain validation used the project directory:

```text
validation_runs/prep_isce_real_full_ps_8x2
```

The run completed through Step 8 with merged outputs including:

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

Generated validation outputs are not committed to Git because of size.

## Basic Usage

Install dependencies in a Python environment:

```bash
pip install -r requirements.txt
```

Prepare ISCE2 stack data:

```bash
python stamps_python/prep_isce.py --help
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

- The Python workflow currently targets PS-InSAR. Small Baseline support is only
  partially translated and should be treated as experimental/incomplete.
- The project is designed around HDF5 outputs (`*.h5`) rather than MATLAB `.mat`
  files for the main Python path.
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
retains the original MATLAB source for reference. Please cite the appropriate
StaMPS publications and respect the GPL-3.0 license when using or distributing
this work.
