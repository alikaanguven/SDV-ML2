# Training

Training runs in the `SDV` Conda environment inside the GPU Apptainer image.
The SLURM wrapper enters this environment automatically:

```bash
ssh clip-gpu-xxx
apptainer exec --nv /software/system/jupyter/jupyter-conda_v10012025.sif bash -l
voms-proxy-init -rfc -voms cms -valid 192:0
cd /groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111
sbatch slurm_scripts/to_gpu.sh "python3 -u training/vtxFramework_train.py"
```

Before submission, set the dataset JSON, output paths, and training parameters
in the training script. A replacement environment needs Python 3.11 with
PyTorch, NumPy, Awkward Array, uproot, Numba, scikit-learn, Matplotlib, and
Neptune when logging is enabled.

# Predicting

Use the same container and `SDV` environment:

```bash
ssh clip-gpu-xxx
apptainer exec --nv /software/system/jupyter/jupyter-conda_v10012025.sif bash -l
mamba activate SDV
cd /groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111
```

No need to require more than 3 cores CPU, and 20G memory, and 1 GPU.

Prediction requires the files to be present somewhere accessable to write fast and easily.
For example `scratch-cbe` is a good choice.
Prediction will write the parquet files in those directories.
Make sure you use only a single core in the precision.


Set `JSON_PATH`, `MODEL_PATH`, `INPUT_BASEDIR`, and the sample filters in the
selected driver. Run other years and 2024 separately:

```bash
nohup python3 -u testing/predict_main.py > testing/predict.log 2>&1 &
nohup python3 -u testing/predict_main24.py > testing/predict24.log 2>&1 &
```

The standard path uses `MET_*` and the stored `Jet_jetId`. The 2024 path uses
`PuppiMET_*` and evaluates the official `AK4PUPPI_TightLeptonVeto` Jet ID with
`correctionlib`. It therefore requires access to `/cvmfs/cms-griddata.cern.ch`.
A replacement environment needs PyTorch, Awkward Array, uproot, PyArrow,
NumPy, Numba, pandas, and correctionlib.

# Adding branches

Back up the ROOT files first. Configure `INPUTBASE_DIR`, `MODEL_NAME`, the JSON,
the year filter, and the SLURM launcher in `testing/addMLvtx_main_v3.py`, then
submit:

```bash
python3 testing/addMLvtx_main_v3.py
squeue --me
sacct
```

This step needs PyROOT, PyArrow, and pandas. Use ROOT `6.28/06` or newer. The
`SDV` environment has ROOT `6.28/00`; do not use it to modify 2024 files. For
2024, select `slurm_scripts/to_cpu1_24.sh`, which enters CMSSW `15_0_5` with
ROOT `6.32.11`. For other years, use `slurm_scripts/to_cpu1_8G.sh` from a shell
with a suitable ROOT environment.

The writer verifies that the prediction count equals `sum(nSDVSecVtx)` before
replacing each file. Check a representative output and any temporary files:

```bash
rootls -t path/to/file.root | grep vtx_PART_1111best_valloss_epoch
python3 testing/check_addMLvtx_temp_files.py INPUTBASE_DIR
```
