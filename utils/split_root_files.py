#!/usr/bin/env python3
# split_roots.py
#
# Usage:
#   python3 split_roots.py --glob "/path/to/**/*.root" --chunk 2000 --outdir /path/to/out
# Optional:
#   --tree Events          (default)
#   --compression 1        (0 fastest, 9 smallest)
#
# Result:
#   For each input file like foo.root -> outdir/foo__split_000.root, foo__split_001.root, ...

import time
import os
import glob
import argparse
import ROOT


def main():
    ROOT.gROOT.SetBatch(True)

    p = argparse.ArgumentParser(description="Split ROOT TTrees into fixed-size chunks.")
    p.add_argument("--glob", required=True, help="Input file glob (quotes recommended).")
    p.add_argument("--chunk", type=int, required=True, help="Chunk size (events per output file).")
    p.add_argument("--outdir", required=True, help="Directory to write split files.")
    p.add_argument("--tree", default="Events", help="TTree name (default: Events).")
    p.add_argument("--compression", type=int, default=1, help="ROOT compression 0..9 (default: 1).")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise SystemExit("No input files matched the glob.")

    os.makedirs(args.outdir, exist_ok=True)

    for i, in_path in enumerate(files):
        f_in = ROOT.TFile.Open(in_path, "READ")
        if not f_in or f_in.IsZombie():
            print(f"[skip] cannot open: {in_path}")
            continue

        t_in = f_in.Get(args.tree)
        if not t_in or not isinstance(t_in, ROOT.TTree):
            print(f"[skip] tree '{args.tree}' not found in: {in_path}")
            f_in.Close()
            continue

        n = int(t_in.GetEntries())
        if n <= 0:
            print(f"[skip] empty tree in: {in_path}")
            f_in.Close()
            continue

        base = f"out_{i:04d}"
        chunk = max(1, args.chunk)

        print(f"[split] {in_path}  entries={n}  -> chunks of {chunk}")

        start = 0
        part  = 0
        while start < n:
            take = min(chunk, n - start)
            out_path = os.path.join(args.outdir, f"{base}__split_{part:03d}.root")

            # Create new output file, ensure subsequent objects go there
            f_out = ROOT.TFile(out_path, "RECREATE", "", args.compression)
            f_out.cd()

            # Fast, robust range copy:
            # CopyTree creates a new tree (same name) in *current* file/dir.
            # Selection on Entry$ avoids per-entry Python loops and keeps branches correct.
            sel = f"(Entry$>={start}) && (Entry$<{start + take})"
            t_out = t_in.CopyTree(sel, "", take, start)

            # Safety: write and close
            if t_out:
                t_out.Write()  # single 'Events' in the file
                n_written = int(t_out.GetEntries())
            else:
                n_written = 0

            f_out.Close()

            print(f"  -> {out_path}  entries={n_written}  (range [{start},{start+take}))")

            start += take
            part  += 1

        f_in.Close()


if __name__ == "__main__":
    main()
