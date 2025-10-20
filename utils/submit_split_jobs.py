import json
import os 
import shutil
from pathlib import Path


import subprocess
from concurrent.futures import ThreadPoolExecutor


# ### SIGNAL


# # Read json file and construct sample patterns
# # ----------------------------------------------------------------------

# json_file = "/groups/hephy/cms/ang.li/MLjson/CustomNanoAOD_MLtraining_20250910.json"
# with open(json_file, "r") as f:
#     data = json.load(f)

# patterns = []
# for key, value in data["CustomNanoAOD_MLtraining_20250910"]["dir"].items():
#     pattern = os.path.join(value, "**/*.root")
#     patterns.append(pattern)


# # Create fresh directory
# OUTBASEDIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/CustomNanoAOD_MLtraining_20250910_split'
# if os.path.exists(OUTBASEDIR): shutil.rmtree(OUTBASEDIR)
# os.mkdir(OUTBASEDIR)

# def run_job(pattern):
#     sample = Path(pattern).parents[2].name
#     outdir = os.path.join(OUTBASEDIR, sample)
#     os.mkdir(outdir)

#     print(f"→ {sample}")
#     subprocess.run([
#         "python3", "utils/split_root_files.py",
#         "--glob", pattern,
#         "--chunk", str(2000),
#         "--outdir", outdir
#     ], check=True)


# MAX_JOBS = 20  # number of parallel subprocesses
# with ThreadPoolExecutor(max_workers=MAX_JOBS) as pool:
#     pool.map(run_job, patterns)


### BKG


BKGDIR = '/scratch-cbe/users/ang.li/SoftDV/NanoAOD_MLNano_v3'
patterns = [f'{BKGDIR}/qcdht0700_2018/**/*.root',
            f'{BKGDIR}/qcdht1000_2018/**/*.root',
            f'{BKGDIR}/qcdht1500_2018/**/*.root',
            f'{BKGDIR}/qcdht2000_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht0100_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht0200_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht0400_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht0600_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht0800_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht1200_2018/**/*.root',
            f'{BKGDIR}/wjetstolnuht2500_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht0100_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht0200_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht0400_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht0600_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht0800_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht1200_2018/**/*.root',
            f'{BKGDIR}/zjetstonunuht2500_2018/**/*.root',
            f'{BKGDIR}/st_tW_tbar_2018/**/*.root',
            f'{BKGDIR}/ttbar_2018/**/*.root',
            ]

# Create fresh directory
OUTBASEDIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/CustomNanoAOD_MLtraining_20250910_split/bkg'
if os.path.exists(OUTBASEDIR): shutil.rmtree(OUTBASEDIR)
os.makedirs(OUTBASEDIR, exist_ok=True)

def run_job(pattern):
    sample = Path(pattern).parents[1].name
    outdir = os.path.join(OUTBASEDIR, sample)
    os.makedirs(outdir, exist_ok=True)

    print(f"→ {sample}")
    subprocess.run([
        "python3", "utils/split_root_files.py",
        "--glob", pattern,
        "--chunk", str(500),
        "--outdir", outdir,
        "--maxchunks", str(1),
        "--maxfiles", str(20),
    ], check=True)


MAX_JOBS = 20  # number of parallel subprocesses
with ThreadPoolExecutor(max_workers=MAX_JOBS) as pool:
    pool.map(run_job, patterns)