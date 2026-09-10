# ML_submitter.py
# Simple grid submitter for Slurm using your to_gpu.sh wrapper.
# Edit GRID below. Run:  python3 ML_submitter.py

import itertools
import shlex
import subprocess


# ---- basic settings ----------------------------------------------------------

PYTHON = "python3"                      # or "python"
SCRIPT = "training/vtx_ABCDiscoTECmdmm_split_train_forgrid.py"          # change if your file name differs
WRAPPER = "slurm_scripts/to_gpu234.sh"                   # your Slurm wrapper that runs "$1"

EXTRA = None # "--verbose"                    # extra flags appended to every run (or "")
DRY_RUN = False                               # set True to only print sbatch commands


# ---- define your grid here ---------------------------------------------------
# Keys are the flags as you’d pass them on the CLI (include the leading --).
# Values are lists of strings (one or many).
GRID = {
    "--init_lr":    ["3e-5", "1e-4", "3e-4"],
    "--alpha_lr":   ["3e-5", "6e-5"],
    "--k":          ["75", "100"],
    "--eps_disco":    ["2e-2", "5e-2", "1e-1"],
}


# ---- machinery (minimal, readable) ------------------------------------------

def build_axes(grid):
    # Preserve the order you wrote in GRID
    axes = []
    for flag, values in grid.items():
        pairs = [(flag, v) for v in values]
        axes.append(pairs)
    return axes


def build_command(pairs):
    # tokens: [python, script, --k1, v1, --k2, v2, ... , EXTRA...]
    tokens = [PYTHON, SCRIPT]

    for flag, val in pairs:
        tokens.append(flag)
        tokens.append(val)

    if EXTRA:
        tokens.extend(shlex.split(EXTRA))

    # one quoted string for to_gpu.sh’s $1
    return " ".join(shlex.quote(t) for t in tokens)


def submit(cmd_str):
    sbatch = ["sbatch", WRAPPER, cmd_str]
    if DRY_RUN:
        print(" ".join(shlex.quote(x) for x in sbatch))
    else:
        subprocess.run(sbatch, check=True)


def main():
    axes = build_axes(GRID)

    # If GRID is empty, still run once using script defaults
    combos = itertools.product(*axes) if axes else [()]

    count = 0
    for combo in combos:
        cmd = build_command(list(combo))
        # print(f"Submitting: {cmd}")
        submit(cmd)
        count += 1
        # break

    print(f"{'[dry-run] ' if DRY_RUN else ''}Total jobs: {count}")


if __name__ == "__main__":
    main()
