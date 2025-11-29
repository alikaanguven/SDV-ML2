#!/usr/bin/env python3
import os
import json
import shutil
import ROOT

# --------------------------------------------------------------------------
#       This script skips all the L1_* and HLT_* branches
#       except for HLT_PFMETNoMu120_PFMHTNoMu120_IDTight!
#       
#       Because when that happens unfornutately those files are skipped!
#       We don't want that.
# 
#       Example warning when a file is skipped due to missing HLT branches:
#       TTree::CopyEntries:0: RuntimeWarning: One of the export top level branches (HLT_Diphoton30_18_PVrealAND_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55) is not present in the import TTree.
#       TTree::CopyEntries:0: RuntimeWarning: Skipped file /scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_nano_data/met2018a/output/out_NANOAODoutput_125.root
#
# --------------------------------------------------------------------------



# HLT branch we want to KEEP even though it starts with "HLT_"
KEEP_HLT_BRANCH = "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight"


def find_root_files_recursive(base_dir):
    """All .root files under base_dir (recursively), excluding merged ones."""
    files = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        # don't descend into any merge dirs if they exist
        dirnames[:] = [d for d in dirnames if not d.startswith("merged_")]
        for fn in filenames:
            if not fn.endswith(".root"):
                continue
            if "merged" in fn:
                continue
            files.append(os.path.join(dirpath, fn))
    return sorted(files)


def chunk_by_size(files, max_bytes):
    """Greedy chunking of file paths into groups with total size ~max_bytes."""
    chunks, cur, cur_size = [], [], 0
    for path in files:
        size = os.path.getsize(path)
        if cur and cur_size + size > max_bytes:
            chunks.append(cur)
            cur, cur_size = [], 0
        cur.append(path)
        cur_size += size
    if cur:
        chunks.append(cur)
    return chunks


def check_input_files(files, tree_name):
    """
    Check for zombie files or missing trees.
    - Prints big warnings.
    - Raises RuntimeError if any zombie/unreadable file is found.
    """
    any_zombie = False
    any_missing = False

    for path in files:
        f = ROOT.TFile.Open(path, "READ")
        if not f or f.IsZombie():
            print("#" * 80)
            print(f"### WARNING: ZOMBIE or unreadable file detected: {path}")
            print("#" * 80)
            any_zombie = True
            if f:
                f.Close()
            continue

        t = f.Get(tree_name)
        if not t:
            print("#" * 80)
            print(f"### WARNING: No TTree '{tree_name}' in file: {path}")
            print("#" * 80)
            any_missing = True

        f.Close()

    if any_zombie:
        print("#" * 80)
        print("### ERROR: One or more zombie/unreadable input files detected. Aborting.")
        print("#" * 80)
        raise RuntimeError("Zombie or unreadable input files detected.")

    if any_missing:
        print("#" * 80)
        print("### WARNING: Some files are missing the requested TTree. Please inspect the log.")
        print("#" * 80)


def pick_template_file(files, tree_name):
    """
    Pick the file with the largest number of non-HLT_/non-L1_ branches
    in the given tree, but count KEEP_HLT_BRANCH as a normal branch.
    Assumes files are already sanity-checked; no zombie checks here.
    """
    best_file = None
    best_n = -1

    for path in files:
        f = ROOT.TFile.Open(path, "READ")
        # We assume no zombies here thanks to check_input_files; minimal guard:
        if not f:
            continue

        t = f.Get(tree_name)
        if not t:
            f.Close()
            continue

        n = 0
        for br in t.GetListOfBranches():
            name = br.GetName()

            # Ignore all L1_* branches
            if name.startswith("L1_"):
                continue

            # Ignore HLT_* except the one we explicitly want to keep
            if name.startswith("HLT_") and name != KEEP_HLT_BRANCH:
                continue

            n += 1

        if n > best_n:
            best_n = n
            best_file = path

        f.Close()

    if best_file is None:
        raise RuntimeError(f"Could not find any valid template file with tree {tree_name}")

    print(f"  template file: {best_file} ({best_n} non-trigger branches)")
    return best_file


def merge_chunk_ignore_triggers(files, out_path, tree_name):
    """
    Merge one chunk of files into out_path using TChain::CloneTree("fast"),
    ignoring all HLT_* and L1_* branches, except KEEP_HLT_BRANCH which is kept.
    """
    tmpl = pick_template_file(files, tree_name)

    chain = ROOT.TChain(tree_name)
    # add template first
    chain.Add(tmpl)
    for fpath in files:
        if fpath == tmpl:
            continue
        chain.Add(fpath)

    n_entries = chain.GetEntries()
    print(f"    chain has {n_entries} entries from {len(files)} files")

    # Disable triggers, keep everything else
    chain.SetBranchStatus("*", 1)
    chain.SetBranchStatus("HLT_*", 0)
    chain.SetBranchStatus("L1_*", 0)
    # Re-enable the one HLT branch we actually want
    chain.SetBranchStatus(KEEP_HLT_BRANCH, 1)

    # Write only to the output path (under destination), never to input dirs
    out_file = ROOT.TFile(out_path, "RECREATE")
    if not out_file or out_file.IsZombie():
        raise RuntimeError(f"Cannot create output file {out_path}")

    out_tree = chain.CloneTree(-1, "fast")
    out_tree.Write()
    out_file.Close()

    # optional: clean "tag" objects (in the *output* file)
    f_out = ROOT.TFile.Open(out_path, "UPDATE")
    if f_out and not f_out.IsZombie():
        f_out.Delete("tag;*")
        f_out.Write()
        f_out.Close()
        print(f"    [cleaned tags] {out_path}")
    else:
        print(f"    [warn] could not open {out_path} for cleaning")


def main(json_path, tree_name, dst_base, chunk_gb):
    with open(json_path) as jf:
        cfg = json.load(jf)

    sample_dirs = cfg["CustomNanoAOD_GNN"]["dir"]  # mapping: key -> input directory
    chunk_bytes = int(chunk_gb * 1024**3)

    for sample, src_dir in sorted(sample_dirs.items()):
        print(f"\n=== {sample} ===")

        if not os.path.isdir(src_dir):
            print(f"  [warn] src_dir {src_dir} does not exist, skipping")
            continue

        files = find_root_files_recursive(src_dir)
        if not files:
            print(f"  [warn] no ROOT files found under {src_dir}, skipping")
            continue

        print(f"  found {len(files)} ROOT files")

        # First step: zombie / missing-tree check with big warnings
        check_input_files(files, tree_name)

        chunks = chunk_by_size(files, chunk_bytes)
        print(f"  → {len(chunks)} merged outputs (~{chunk_gb} GiB input each)")

        # final destination dir per sample
        dst_dir = os.path.join(dst_base, sample)
        os.makedirs(dst_dir, exist_ok=True)

        # staging dir lives under the destination, not under source
        local_merge_dir = os.path.join(dst_dir, "merged_chain_tmp")
        if os.path.isdir(local_merge_dir):
            print(f"  removing old staging dir: {local_merge_dir}")
            shutil.rmtree(local_merge_dir)
        os.makedirs(local_merge_dir, exist_ok=True)

        for i, chunk in enumerate(chunks):
            out_name = (
                f"{sample}__merged.root"
                if len(chunks) == 1
                else f"{sample}__merged_{i:03d}.root"
            )
            local_out = os.path.join(local_merge_dir, out_name)
            dst_out   = os.path.join(dst_dir,       out_name)

            print(f"  chunk {i+1}/{len(chunks)}")
            merge_chunk_ignore_triggers(chunk, local_out, tree_name)

            shutil.move(local_out, dst_out)
            print(f"    -> {out_name}  (copied to {dst_dir})")

        print(f"  done, removing staging dir {local_merge_dir}")
        shutil.rmtree(local_merge_dir)


if __name__ == "__main__":
    JSON_PATH = "jsons/CustomNanoAOD_GNN.json"
    TREE_NAME = "Events"

    DST_BASE  = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_nano_data_merged"
    CHUNK_GB  = 4.0

    main(
        JSON_PATH,
        TREE_NAME,
        DST_BASE,
        CHUNK_GB,
    )
