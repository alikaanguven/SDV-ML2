#!/usr/bin/env python3
import os, sys, ROOT, glob

def count_events(path):
    files = sorted(glob.glob(os.path.join(path, "*.root")))
    if not files:
        print("No ROOT files found.")
        return

    grand = 0
    for fpath in files:
        f = ROOT.TFile.Open(fpath)
        if not f or f.IsZombie():
            print(f"[ERROR] cannot open {fpath}")
            continue

        t = f.Get("Events")
        n = t.GetEntries() if t else 0
        f.Close()

        grand += n
        print(f"{os.path.basename(fpath):40s}  Events: {n:,}")

    print("\nTOTAL Events:", f"{grand:,}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: count_events.py <directory>")
        sys.exit(1)
    count_events(sys.argv[1])
