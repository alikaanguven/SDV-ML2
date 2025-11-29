#!/usr/bin/env python3
import os, glob, subprocess, shutil
import ROOT

# --------------------------------------------------------------------------
#       WARNING: This script might skip files if there are any incompatible HLT/L1 branches!
#       More info:
#       1. https://root-forum.cern.ch/t/hadd-not-working-when-missing-branches-in-one-of-the-root-files/62957
#       2. https://github.com/root-project/root/pull/17650
#       
#       Example warning when a file is skipped due to missing branches:
#       TTree::CopyEntries:0: RuntimeWarning: One of the export top level branches (HLT_Diphoton30_18_PVrealAND_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55) is not present in the import TTree.
#       TTree::CopyEntries:0: RuntimeWarning: Skipped file /scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_nano_data/met2018a/output/out_NANOAODoutput_125.root
#
# --------------------------------------------------------------------------




SRC_BASE   = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_nano_data"
DST_BASE   = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_nano_data_merged"

CHUNK_GB   = 4.0
CHUNK_BYTES= int(CHUNK_GB * 1024**3)
HADD_J     = "1"   # threads for hadd


for sample in sorted(os.listdir(SRC_BASE)):
    src_dir = os.path.join(SRC_BASE, sample)
    if not os.path.isdir(src_dir):
        continue
    print(f"Processing sample: {sample}")
    files = sorted(f for f in glob.glob(os.path.join(src_dir, "*.root"))
                   if "merged" not in os.path.basename(f))
    if not files:
        continue # make output dirs
    
    print(files)
    # make output dirs
    local_merge_dir = os.path.join(src_dir, "merged_4GB")
    if os.path.isdir(local_merge_dir):
        print("Warning: Deleting existing local merge dir: ", local_merge_dir)
        shutil.rmtree(local_merge_dir)
    os.makedirs(local_merge_dir, exist_ok=True)

    dst_dir = os.path.join(DST_BASE, sample)
    os.makedirs(dst_dir, exist_ok=True)


    # greedy chunking to ~4 GiB
    chunks, cur, cur_size = [], [], 0
    for f in files:
        s = os.path.getsize(f)
        if cur and cur_size + s > CHUNK_BYTES:
            chunks.append(cur)
            cur, cur_size = [], 0
        cur.append(f)
        cur_size += s
    if cur:
        chunks.append(cur)

    print(f"{sample}: {len(files)} files → {len(chunks)} merged outputs (~{CHUNK_GB} GiB)")

    # merge and copy
    for i, chunk in enumerate(chunks):
        out_name = f"{sample}__merged.root" if len(chunks) == 1 else f"{sample}__merged_{i:03d}.root"
        local_out = os.path.join(local_merge_dir, out_name)
        dst_out   = os.path.join(dst_dir, out_name)

        cmd = ["hadd", "-fkT", local_out] + chunk
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(f"  !! hadd failed ({out_name}) with code {res.returncode}")
            continue
        
        # delete tags ...
        f = ROOT.TFile.Open(local_out, "UPDATE")
        if f and not f.IsZombie():
            f.Delete("tag;*")
            f.Write()   # persist metadata changes
            f.Close()
            print(f"[cleaned tags] {local_out}")
        else:
            print(f"[warn] could not open {local_out} for cleaning")


        shutil.copy2(local_out, dst_out)
        os.remove(local_out)
        print(f"  -> {out_name}  (copied to {dst_dir})")
    
    print("Merging finished. Deleting local merge dir: ", local_merge_dir)
    shutil.rmtree(local_merge_dir)