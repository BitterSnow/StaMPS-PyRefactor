# Validation Baseline - 2026-05-09

This note records the first validation pass for the MATLAB-to-Python StaMPS
translation. It is a working baseline, not a final acceptance report.

## Tooling

Added `stamps_python/validate_outputs.py` to compare MATLAB `.mat` reference
files with Python `.h5` outputs.

The tool reports shape, dtype summary, finite/NaN/Inf counts, sampled absolute
differences, and phase differences for complex arrays. MATLAB v7.3 files are
read through HDF5.

Example:

```bash
D:\env\python311\python.exe stamps_python\validate_outputs.py ^
  test_data\PATCH_437\ps1.mat test_data\PATCH_437\ps1.h5 ^
  --keys n_ps n_ifg master_ix master_day day lonlat xy ij bperp
```

## Current Findings

Step 1 data loading is effectively aligned for both `PATCH_437` and
`PATCH_438`.

- `n_ps`, `n_ifg`, `master_ix`, `master_day`, `day`, `ij`, `lonlat`, `bperp`
  match the MATLAB reference.
- `xy` differs only at float precision level, with maximum sampled difference
  around `0.0013`.

Step 2 gamma estimation is close but not bit-identical.

- `K_ps`, `C_ps`, and `coh_ps` show small sampled differences.
- `Nr` random distribution differs more noticeably, which can shift Step 3
  thresholding.
- Adjusted random phase simulation to keep MATLAB-like double precision until
  `exp(1j * rand_ifg)`, instead of forcing `float32`/`complex64` early.

Step 3 is the first clear divergence point.

- `PATCH_437`: MATLAB `select1.mat` has `49971` selected candidates; Python
  `select1.h5` has `49919`.
- `PATCH_438`: MATLAB has `63817`; Python has `63846`.
- `ifg_index` matches.
- Threshold-related outputs differ; this needs a focused pass against
  `ps_select.m`.
- Fixed one Step 3 bug: Python `D_A` bins now match MATLAB
  `D_A_sort(bin_size:bin_size:end-bin_size)`. Before the fix Python produced
  one extra bin, reducing `max_percent_rand`.
- With MATLAB `pm1.mat`/`da1.mat` as input, the Python threshold function now
  reproduces MATLAB `coh_thresh`, `coh_thresh_coeffs`, and initial `ix` for
  both patches. Remaining current-output differences are therefore upstream
  from Step 2 (`pm1.h5`) and from already-generated stale `select1.h5` files.
- Fixed `select1.h5:/ix` output type from `uint16` to `int32`; large patches can
  exceed 65535 candidates.
- Added Matlab v7.3 compatibility in the `reest_flag=2` path: `ph_patch2` is
  converted from compound real/imag to complex and transposed from MATLAB
  `(n_ifg, n_ps)` layout to Python `(n_ps, n_ifg)` layout.
- Performance pass:
  - Step 3 re-estimation now caches `ph_patch2` by `(grid_i, grid_j)`. Current
    Python behavior zeros the whole grid cell before filtering, so all selected
    candidates in the same cell share the same filtered patch value.
  - `PATCH_437` full Step 3 re-estimation completed in the validation copy.
    Patch re-estimation took `238.7 s` for `49969` candidates, with `7639`
    unique grid cells and `84.7%` cache hits. Before caching the same run
    exceeded the 15-minute timeout.
  - Step 2 now skips intermediate `pm1.h5` writes and saves only the final
    iteration because restart loading is not implemented.
  - Step 2 `pm1.h5` now skips restart-only state (`ph_weight`,
    `coh_ps_save`, `weighting`) by default. Set
    `STAMPS_SAVE_PM_RESTART_STATE=1` to retain those arrays for diagnostics.
    On `PATCH_437`, this reduces `pm1.h5` from about `113.8 MB` to
    `71.9 MB`, although HDF5 write time remains dominated by the remaining
    large arrays in this environment.
  - Large Step 2/3 arrays are written uncompressed to avoid compression CPU
    dominating runtime.
  - Step 3 `select1.h5` now skips the large `ph_patch2` and `ph_res2` datasets
    by default. On this machine, writing `ph_patch2`/`ph_res2` cost roughly
    `50-100 s`; metadata-only Step 3 output for a `PATCH_437`-sized selection
    writes in about `0.057 s` and is about `1.39 MB`.
  - Set `STAMPS_SAVE_SELECT_PH_RES2=1` to keep residuals for downstream
    diagnostics. Set both `STAMPS_SAVE_SELECT_PH_PATCH2=1` and
    `STAMPS_SAVE_SELECT_PH_RES2=1` when a Python `select1.h5` must support
    `reest_flag=2` reuse.
- Non-destructive validation was run under
  `validation_runs/step23_20260509`.
  - `PATCH_437` with `reest_flag=2`: `ix`, `keep_ix`, `K_ps2`, `C_ps2`,
    `coh_ps2`, and `ifg_index` match MATLAB.
  - `PATCH_438` with `reest_flag=2`: `ix`, `K_ps2`, `C_ps2`, `coh_ps2`, and
    `ifg_index` match MATLAB. One `keep_ix` differs because the quick test
    intentionally mixed MATLAB `K_ps2` with Python `pm1.h5:/K_ps`; that one
    point sits across the `abs(K_ps - K_ps2) < 2*pi/bperp_range` boundary.

Current Step-2 diagnosis:

- `ps_topofit_batch` is aligned. Recomputing `K_ps`, `C_ps`, `coh_ps`, and
  `ph_res` from MATLAB `ph_patch`/`bperp_mat` matches MATLAB within about
  `1e-6`.
- `clap_filt` is aligned. Filtering MATLAB `ph_grid` and extracting
  `ph_patch` matches MATLAB within about `2.5e-7`.
- The remaining Step-2 differences come from the iterative chain that builds
  `ph_weight`/`ph_grid`, seeded by random-distribution and weighting
  differences. With the current code, regenerating `PATCH_437 pm1.h5` gives
  Step-3 initial selection of `49969` PS versus MATLAB `49971`.
- Cross-checking `PATCH_437` shows replacing Python `Nr` with MATLAB `Nr`
  changes initial selection from `49969` to `49970`; replacing Python
  `coh_ps` with MATLAB `coh_ps` gives MATLAB's `49971`. The remaining
  candidate mismatch is therefore mostly coherence/iteration related, not the
  random histogram alone.
- Regenerating `PATCH_438 pm1.h5` with the current code gives Step-3 initial
  selection of `63805` PS versus MATLAB `63817`. Replacing Python `Nr` with
  MATLAB `Nr` still gives `63805`; replacing Python `coh_ps` gives MATLAB's
  `63817`, confirming the same source of mismatch.
- Most changed candidates sit within about `1e-3` of the Step-3 threshold.
  One `PATCH_438` point (`39029`) has a larger `coh_ps` jump
  (`+0.0776`), but its `ph_grid` cell and `ph_patch` differ only slightly
  (`ph_patch` phase max about `0.0084 rad`). The jump is caused by low-coherence
  topofit peak sensitivity rather than an obvious indexing or CLAP mismatch.

Step 4 and Step 5 inherit the Step 3 divergence.

- `PATCH_437` final `ps2` point count: MATLAB `49573`, Python `49457`.
- `PATCH_438` final `ps2` point count: MATLAB `63271`, Python `63144`.
- Date and baseline metadata still match.
- Root-level `test_data/ps2.mat` has `1303071` PS points while
  `test_data/ps2.h5` has `84435`. These are not a direct pair for final
  Step 6 numeric comparison.

Step 4 was isolated with MATLAB Step-3 inputs under
`validation_runs/step45_20260509/isolate_step4`.

- `PATCH_437`: Python Step 4 produced `49573` PS from MATLAB `49971`
  candidates, matching MATLAB.
- `PATCH_438`: Python Step 4 produced `63271` PS from MATLAB `63817`
  candidates, matching MATLAB.
- For both patches, isolated `ps2`, `pm2`, `ph2`, and `bp2` match MATLAB
  exactly after accounting for MATLAB/Python transpose conventions.
- `weed1.ix_weed`, `weed1.ix_weed2`, and `weed1.ifg_index` match exactly.
  `ps_std`/`ps_max` differ only at float rounding level
  (`~1e-6` for `ps_std`, `~1e-4` for `ps_max`).

Step 5a was isolated on those Step-4 outputs.

- `ph_reref` matches MATLAB exactly for both patches.
- `ph_rc` phase differences are at `~1.2e-7 rad`; absolute complex differences
  are driven by representation/magnitude details, not phase disagreement.

Step 5b merge was run in
`validation_runs/step45_20260509/isolate_step5_merge` using the two isolated
patch outputs.

- Merge completed with `84618` PS and wrote root `ps2`, `pm2`, `ph2`, `bp2`,
  `rc2`, `hgt2`, `inc2`, and `ifgstd2`.
- No strict MATLAB reference exists for this two-patch isolated merge in the
  current data tree. The root MATLAB `ps2.mat` is a larger full-project result
  (`1303071` PS), so it is not a valid numerical reference for this run.
- Existing root `test_data/*.h5` outputs are stale relative to the latest fixes:
  root `test_data/ps2.h5` still has `day_ix` as `0:106`, while newly generated
  outputs now save MATLAB-compatible `1:107`.

Validation fixes made during this pass:

- `load_mat` no longer decodes uint8 MATLAB v7.3 arrays as UTF-16 strings; this
  fixes `select1.mat:/keep_ix` loading.
- `load_mat` now converts MATLAB v7.3 compound complex datasets with
  `real`/`imag` fields into NumPy complex arrays.
- Step-1, Step-4, and Step-5 metadata saving now preserves MATLAB-compatible
  1-based `day_ix`.
- Step-5 merge now treats an empty `pm2.h5:/ph_res` dataset as missing. This is
  needed when Step 3 is run in fast-save mode and Step 4 carries an empty
  residual array forward.

Current end-to-end two-patch Python output:

- `validation_runs/step2_current_20260509` was rebuilt from current Python
  Step 2/3 outputs through Step 4, Step 5a, Step 5b, and Step 6.
- Step 4 point counts from the current Python Step 3 fast path:
  - `PATCH_437`: `49969 -> 49572`
  - `PATCH_438`: `63805 -> 63260`
- Step 5b merge completed with `84607` PS and wrote root `ps2`, `pm2`, `ph2`,
  `bp2`, `rc2`, `hgt2`, `inc2`, and `ifgstd2`.
- Step 6 input assembly checks passed:
  - `ph_w`: `(84607, 108)` unit complex, no zero entries.
  - `unwrap_ifg_index`: `107` IFGs after excluding master image index `41`.
  - `ifgday_ix`: `(108, 2)` and follows non-SB `[master_ix, 1:n_ifg]`.
  - `day - master_day`: finite, range `[-504, 828]`.
  - `bperp_mat`: `(84607, 108)`.
- Current Step 6 ran successfully and wrote `phuw2.h5`.
  - `ph_uw`: `(84607, 108)`, finite, master column zero.
  - `msd`: `(108,)`, finite, range approximately `0` to `10.97`.
  - Internal wrapped-phase consistency check passed:
    `angle(exp(1j*(ph_uw - angle(ph_w))))` has max residual about
    `5.9e-7 rad`.
  - `ph_uw - angle(ph_w)` is an integer-cycle field to numerical precision:
    max error to nearest integer cycle is about `9.4e-8`.
  - Snaphu intermediate files are dimensionally consistent with a `32 x 32`
    grid: `snaphu.in` is `8192` bytes, `snaphu.out` is `4096` bytes, and
    `snaphu.costinfile` is `15872` bytes.
  - `validation_runs/step2_current_20260509/step6_internal_check.json` records
    the detailed internal consistency statistics.
  - No direct MATLAB numeric acceptance was performed for this two-patch run
    because the available root MATLAB reference is a much larger full-project
    result, not this isolated two-patch dataset.

Current Step-7 health check:

- `scla_estimation.py` now initialises `StampsConfig` with the target
  `patch_dir`, so validation runs outside the repository root can load their
  local `parms.mat`.
- Empty MATLAB drop-index placeholders such as `array([0, 0], dtype=uint64)`
  are treated as empty in Step 7. This prevents an unintended extra IFG drop.
- Step-7 input assembly on `validation_runs/step2_current_20260509` passed:
  - `ph_uw`: `(84607, 108)`.
  - `bperp_mat`: `(84607, 107)`, expanded to `(84607, 108)` with master column.
  - non-master unwrap IFGs: `107`; sequential observations: `106`.
  - design matrix `G`: `(106, 3)`, rank `3`.
  - `ifg_vcm`: `(108, 108)` from `ifgstd2.h5`.
- Current Step 7 ran successfully and wrote `scla2.h5` and
  `scla_smooth2.h5`.
  - all `K_ps_uw`, `C_ps_uw`, and `ph_scla` arrays are finite.
  - `ph_scla` matches `K_ps_uw * bperp_mat` to about `1e-7`.
  - master column of `ph_scla` is zero.
  - smoothing clipped `17870` K values and `19997` C values in this two-patch
    run.
- `validation_runs/step2_current_20260509/step7_input_check.json` and
  `step7_output_check.json` record the detailed Step-7 statistics.

Current Step-8 health check:

- `scn_filt.py` now initialises `StampsConfig` with the target `patch_dir`, so
  standalone Step-8 validation loads the run-local `parms.mat`.
- Empty MATLAB drop-index placeholders such as all-zero arrays are treated as
  empty in Step 8, matching the Step-7 parser fix.
- SCLA subtraction now indexes `ph_scla` and `ph_ramp` with
  `unwrap_ifg_index` when the SCLA arrays contain full IFG columns. This keeps
  MATLAB behaviour when `drop_ifg_index` is non-empty.
- Step-8 input assembly on `validation_runs/step2_current_20260509` passed:
  - `ph_uw`: `(84607, 108)`.
  - `scla2.ph_scla`: `(84607, 108)`.
  - `scla2.C_ps_uw`: `(84607,)`.
  - `drop_ifg_index`: empty; `unwrap_ifg_index`: `108` IFGs.
  - master column: `40` in 0-based Python indexing.
  - `scn_time_win`: `365` days; `scn_wavelength`: `100` m.
- Current Step 8 ran successfully and wrote `scn2.h5`.
  - Delaunay edge count: `253789`.
  - runtime for this two-patch run was about `200` seconds.
  - `ph_scn_slave`: `(84607, 108)`, finite, `float32`.
  - `ph_hpt`: `(84607, 108)`, finite, `float32`.
  - `ph_ramp`: `(84607, 0)` because `scn_deramp_ifg` is empty.
  - master column of `ph_scn_slave` is exactly zero.
  - first PS row is exactly zero after re-referencing.
  - `std(ph_hpt - ph_scn_slave)` is about `1.068`.
- `validation_runs/step2_current_20260509/step8_input_check.json` and
  `step8_output_check.json` record the detailed Step-8 statistics.

Integrated Steps 1-8 rerun:

- A clean input-only validation directory was created at
  `validation_runs/integrated_20260511_1138` from `test_data` root metadata and
  the two patch candidate input files. Existing `.h5/.mat` processing outputs
  were not copied into the new directory.
- `stamps_main.py --start 1 --end 8 --config validation_runs/integrated_20260511_1138`
  completed successfully through the integrated runner.
- During the first integrated pass, Step 3 exposed one remaining parser issue:
  all-zero MATLAB `drop_ifg_index` placeholders were still treated as real
  entries by `ps_selection_final.py` and `ps_weeding.py`.
- The parser was fixed in both files, and the same integrated directory was
  rerun from Step 3 through Step 8. The rerun completed successfully and no
  longer logged false IFG drops.
- Final integrated counts:
  - `PATCH_437`: Step 1 `50468` candidates, Step 3 `50024` initially selected,
    Step 4 `49539` kept.
  - `PATCH_438`: Step 1 `65193` candidates, Step 3 `64016` initially selected,
    Step 4 `63263` kept.
  - merged root: `84615` PS, `108` IFGs.
- Final root outputs passed structural checks:
  - `ph2`: `(84615, 108)`.
  - `phuw2.ph_uw`: `(84615, 108)`, finite, master column exactly zero.
  - `scla2.ph_scla`: `(84615, 108)`.
  - `scla_smooth2.ph_scla`: `(84615, 108)`.
  - `scn2.ph_scn_slave`: `(84615, 108)`, finite, master column exactly zero,
    first PS row exactly zero after re-referencing.
  - merged `ps2.day_ix` remains MATLAB-compatible 1-based, range `1..107`.
- `snaphu` was invoked during integrated Step 6. The final files are
  dimensionally consistent with a `32 x 32` grid:
  - `snaphu.in`: `8192` bytes.
  - `snaphu.out`: `4096` bytes.
  - `snaphu.costinfile`: `15872` bytes.
  - `snaphu.log` reports `snaphu v1.4.2` and `Program snaphu done`.
- `validation_runs/integrated_20260511_1138/integrated_run_check.json` records
  the final integrated-run structural statistics.

## Flow Fix

`StampsRunner.run()` was adjusted so that:

- Steps 1-5a run per patch.
- Step 5b merge runs once at project level.
- Steps 6-8 run once at project level after merge.

This matches the MATLAB `stamps.m` flow for multi-patch projects. The previous
runner dispatched Steps 6-8 inside each patch directory.

## Next Target

The current Python two-patch validation run now covers Steps 1-8 end-to-end.
Recommended next targets:

- Run the full `StampsRunner` Steps 1-8 on a fresh validation directory to make
  sure the integrated flow reproduces the manually validated sequence.
  Completed in `validation_runs/integrated_20260511_1138`.
- Add a compact regression test around the two-patch run metadata and key array
  invariants, especially `day_ix`, `ifgday_ix`, Step-7/8 shapes, and master
  column zeroing.
- If strict numerical Step-6/7/8 acceptance is needed, generate a matching
  two-patch MATLAB reference. The existing root MATLAB reference in `test_data`
  is a full-project result and is not directly comparable to this two-patch
  validation run.
