# StaMPS Python Full-Chain Runbook

This document describes how to run a new ISCE dataset through the Python
preprocessor and StaMPS Steps 1-8.

The current recommended flow is:

```text
ISCE products
  -> prep_isce.py --write-ps1
  -> PATCH_*/ps1.h5
  -> stamps_main.py --start 1 --end 8
```

`stamps_main.py` Step 1 now reuses existing `PATCH_*/ps1.h5`, so starting from
Step 1 is safe after preprocessing. It will not overwrite `ps1.h5`.

## 1. Input Layout

Use one project directory for the raw/prepared ISCE inputs. In examples below:

```powershell
$SRC = "D:\data\my_isce_project"
$OUT = "D:\data\my_stamps_python_run"
```

The first supported layout is the StaMPS/ISCE-style layout used by
`mt_prep_isce`:

```text
my_isce_project/
  width.txt
  len.txt
  day.1.in
  master_day.1.in
  bperp.1.in
  heading.1.in
  lambda.1.in
  parms.mat
  lon.raw
  lat.raw
  dem.raw
  pscphase.in
  master_YYYYMMDD/master/master.slc        or master/master.slc
  YYYYMMDD/slave.slc
  YYYYMMDD/isce_minrefdem.int
```

`prep_isce.py` currently expects these root metadata files to already exist:

- `width.txt`: raster width in pixels.
- `len.txt`: raster length in pixels.
- `day.1.in`: one slave date per line, `YYYYMMDD`.
- `master_day.1.in`: master date, `YYYYMMDD`.
- `bperp.1.in`: one scalar perpendicular baseline per slave date.
- `heading.1.in`: satellite heading.
- `lambda.1.in`: radar wavelength in meters.
- `parms.mat`: StaMPS parameter file used by the processing pipeline.

The phase list file can be named anything, but this runbook uses
`pscphase.in`. It should contain width on the first line, followed by one
complex interferogram path per slave:

```text
12345
D:\data\my_isce_project\20200113\isce_minrefdem.int
D:\data\my_isce_project\20200125\isce_minrefdem.int
D:\data\my_isce_project\20200206\isce_minrefdem.int
```

For PS mode, the interferogram list should contain slave interferograms only;
do not include a master/master interferogram.

## 2. Quick Sanity Check

From the repository root:

```powershell
cd D:\coding\Stamps_Refactor_Project
python stamps_python\tests\prep_isce_synthetic_check.py
```

Expected result:

```json
"passed": true
```

This confirms the preprocessor, `--write-ps1`, and Step-1 reuse behavior are
working in the current environment.

## 3. Preprocess New Dataset

Run from the repository root:

```powershell
cd D:\coding\Stamps_Refactor_Project

python stamps_python\prep_isce.py `
  "D:\data\my_isce_project" `
  --output "D:\data\my_stamps_python_run" `
  --da-thresh 0.4 `
  --range-patches 1 `
  --azimuth-patches 1 `
  --range-overlap 50 `
  --azimuth-overlap 50 `
  --run-calamp `
  --run-select `
  --run-extract `
  --write-ps1 `
  --lon "D:\data\my_isce_project\lon.raw" `
  --lat "D:\data\my_isce_project\lat.raw" `
  --dem "D:\data\my_isce_project\dem.raw" `
  --ifg-list "D:\data\my_isce_project\pscphase.in" `
  --ifg-list-has-width
```

Typical PS value:

```powershell
--da-thresh 0.4
```

For denser exploratory runs, use a larger threshold such as:

```powershell
--da-thresh 0.6
```

For multiple patches, adjust these:

```powershell
--range-patches 2 --azimuth-patches 3 --range-overlap 50 --azimuth-overlap 200
```

The preprocessor writes:

```text
my_stamps_python_run/
  patch.list
  calamp.in
  calamp.out
  copied metadata files
  PATCH_1/
    patch.in
    patch_noover.in
    pscands.1.ij
    pscands.1.ij.int
    pscands.1.ij0
    pscands.1.da
    pscands.1.ll
    pscands.1.hgt
    pscands.1.ph
    ps1.h5
    no_ps_info.h5
```

## 4. Check Preprocessor Output

Verify that each patch has a non-empty `ps1.h5`:

```powershell
Get-ChildItem "D:\data\my_stamps_python_run\PATCH_*" -Filter ps1.h5 |
  Select-Object FullName,Length,LastWriteTime
```

Optional Python check:

```powershell
python - <<'PY'
from pathlib import Path
import h5py
root = Path(r"D:\data\my_stamps_python_run")
for ps1 in sorted(root.glob("PATCH_*/ps1.h5")):
    with h5py.File(ps1, "r") as f:
        print(ps1.parent.name, "n_ps=", int(f["n_ps"][()]), "ph=", f["ph"].shape)
PY
```

Expected:

- every patch has `n_ps > 0`
- `ph` has shape `(n_ps, n_ifg)`

## 5. Run StaMPS Python Steps 1-8

Run:

```powershell
cd D:\coding\Stamps_Refactor_Project

python stamps_python\stamps_main.py `
  --start 1 `
  --end 8 `
  --config "D:\data\my_stamps_python_run" `
  --log-level INFO
```

Because preprocessing already wrote `PATCH_*/ps1.h5`, Step 1 should log:

```text
Step 1 output already exists for PATCH_x (... PS); reusing ps1.h5
```

Then the runner continues through:

- Step 2: gamma/coherence estimation
- Step 3: PS selection
- Step 4: weeding
- Step 5: phase correction and patch merge
- Step 6: phase unwrapping with `snaphu`
- Step 7: SCLA estimation
- Step 8: SCN filtering

## 6. Resume or Rerun

If a run stops after Step 1 preprocessing, continue from Step 2:

```powershell
python stamps_python\stamps_main.py `
  --start 2 `
  --end 8 `
  --config "D:\data\my_stamps_python_run" `
  --log-level INFO
```

If a run stops after Step 5 merge, continue from Step 6:

```powershell
python stamps_python\stamps_main.py `
  --start 6 `
  --end 8 `
  --config "D:\data\my_stamps_python_run" `
  --log-level INFO
```

To auto-resume based on detected stage for per-patch steps:

```powershell
python stamps_python\stamps_main.py `
  --start 0 `
  --end 8 `
  --config "D:\data\my_stamps_python_run" `
  --log-level INFO
```

For now, explicit `--start` values are easier to reason about during validation.

## 7. Output Files to Inspect

After a successful run, project-root outputs should include:

```text
my_stamps_python_run/
  ps2.h5
  pm2.h5
  ph2.h5
  bp2.h5
  rc2.h5
  hgt2.h5
  inc2.h5
  ifgstd2.h5
  phuw2.h5
  scla2.h5
  scla_smooth2.h5
  scn2.h5
  snaphu.log
```

Each patch should include:

```text
PATCH_*/
  ps1.h5
  pm1.h5
  select1.h5
  weed1.h5
  ps2.h5
  pm2.h5
  ph2.h5
  rc2.h5
```

## 8. Final Structural Check

Run this after Steps 1-8:

```powershell
python - <<'PY'
from pathlib import Path
import h5py
import numpy as np

root = Path(r"D:\data\my_stamps_python_run")

with h5py.File(root / "ps2.h5", "r") as ps:
    n_ps = int(np.asarray(ps["n_ps"]).ravel()[0])
    n_ifg = int(np.asarray(ps["n_ifg"]).ravel()[0])
    day = np.asarray(ps["day"]).ravel()
    master_day = float(np.asarray(ps["master_day"]).ravel()[0])
    master_col = int(np.where(day == master_day)[0][0])
    print("ps2:", n_ps, "PS,", n_ifg, "IFG, master_col=", master_col)

with h5py.File(root / "phuw2.h5", "r") as phuw:
    ph_uw = np.asarray(phuw["ph_uw"])
    print("phuw2:", ph_uw.shape, "finite=", np.isfinite(ph_uw).all())
    print("phuw master max abs:", float(np.max(np.abs(ph_uw[:, master_col]))))

with h5py.File(root / "scn2.h5", "r") as scn:
    ph_scn = np.asarray(scn["ph_scn_slave"])
    print("scn2:", ph_scn.shape, "finite=", np.isfinite(ph_scn).all())
    print("scn master max abs:", float(np.max(np.abs(ph_scn[:, master_col]))))
PY
```

Expected:

- `phuw2` and `scn2` shapes match `(n_ps, n_ifg)`
- all arrays are finite
- master column max absolute value is `0.0` or near numerical zero

Check `snaphu`:

```powershell
Get-Content "D:\data\my_stamps_python_run\snaphu.log"
```

Expected ending:

```text
Program snaphu done
```

## 9. Current Limitations

The current preprocessor is validated for PS-mode, complex float32 inputs.

Supported now:

- complex float32 SLCs and interferograms
- optional byte-swap handling in helper functions
- optional mask for calamp/select
- scalar baseline file `bperp.1.in`
- `--write-ps1` using the existing `ISCEPSLoader` path

Not yet fully automated:

- extracting all metadata directly from arbitrary ISCE XML/VRT layouts
- small-baseline preprocessing
- direct HDF5 writing without legacy `pscands.1.*` intermediates
- baseline grid auto-discovery during preprocessing

For a new dataset, the safest path is to prepare the root metadata text files
explicitly, then run the commands above.
