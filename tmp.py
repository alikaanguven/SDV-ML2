from pathlib import Path

base = Path("/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/IVF_MC_2022_merged")

for d in base.iterdir():
    if d.is_dir():
        parquet_files = list(d.glob("*.parquet"))
        if not parquet_files:
            print(d)