#!/usr/bin/env python3
import os, glob, subprocess, random
import ROOT

# --- Config ---
# SRC_BASE     = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_signal_split"
# DST_BASE     = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_signal_mixed"

SRC_BASE     = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/CustomNanoAOD_MLtraining_20250910_split"
DST_BASE     = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/CustomNanoAOD_MLtraining_20250910_mixed"

# Collect files by NAME pattern (recursive under SRC_BASE)
# e.g. "*.root" or "*__split_*.root" or "stop_*_2018_*.root"
NAME_PATTERN = "*.root"

# OUT_PREFIX   = "stop_mix"   # prefix for output files: e.g. ALL__merged_000.root
OUT_PREFIX   = "train_mix"   # prefix for output files: e.g. ALL__merged_000.root
CHUNK_GB     = 2.0
CHUNK_BYTES  = int(CHUNK_GB * 1024**3)
HADD_THREADS = 1       # -j for hadd
SHUFFLE      = True

os.makedirs(DST_BASE, exist_ok=True)

# --- Gather files (recursively) matching NAME_PATTERN ---
all_files = [
    f for f in glob.glob(os.path.join(SRC_BASE, "**", NAME_PATTERN), recursive=True)
    if os.path.isfile(f)
]
# Exclude previously merged outputs or anything with 'merged' in the basename
all_files = [f for f in all_files if "merged" not in os.path.basename(f)]

if not all_files:
    print("[info] No files matched the pattern. Nothing to do.")
    raise SystemExit(0)

if SHUFFLE:
    random.shuffle(all_files)

# --- Greedy chunking to ~CHUNK_GB each ---
chunks, cur, cur_sz = [], [], 0
for f in all_files:
    s = os.path.getsize(f)
    if cur and cur_sz + s > CHUNK_BYTES:
        chunks.append(cur)
        cur, cur_sz = [], 0
    cur.append(f)
    cur_sz += s
if cur:
    chunks.append(cur)

print(f"Matched {len(all_files)} files → {len(chunks)} merged outputs (~{CHUNK_GB} GiB each)")

# --- Merge per chunk with hadd, clean 'tag' keys, atomic rename ---
for i, chunk in enumerate(chunks):
    out_name = f"{OUT_PREFIX}__merged.root" if len(chunks) == 1 else f"{OUT_PREFIX}__merged_{i:03d}.root"
    final_out = os.path.join(DST_BASE, out_name)
    tmp_out   = final_out + ".tmp"

    # hadd command
    cmd = ["hadd", "-fkT", "-j", str(HADD_THREADS), tmp_out] + chunk
    print(f"[hadd] {out_name} with {len(chunk)} inputs")
    res = subprocess.run(cmd)

    if res.returncode != 0:
        print(f"  !! hadd failed ({out_name}) with code {res.returncode}")
        # Clean up temp file if created
        if os.path.exists(tmp_out):
            try: os.remove(tmp_out)
            except OSError: pass
        continue

    # Delete 'tag' keys, persist, then atomically move into place
    f = ROOT.TFile.Open(tmp_out, "UPDATE")
    if f and not f.IsZombie():
        f.Delete("tag;*")
        f.Write()
        f.Close()
        print(f"  [cleaned tags] {tmp_out}")
    else:
        print(f"  [warn] could not open {tmp_out} for cleaning")

    # Atomic move into final path
    os.replace(tmp_out, final_out)
    print(f"  -> {final_out}")
