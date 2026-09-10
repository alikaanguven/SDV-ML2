#!/usr/bin/env python3

import argparse
import json
import re
import sys
from subprocess import run


HELP_S = """All the jobs starting after this time/date will be searched.
The argument will be passed to sacct.
Pass the date-time like this: 2024-11-14T00:00:00"""

ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resubmit failed addMLvtx jobs from a job JSON."
    )
    parser.add_argument("jobjson", help="Full path to the job json file")
    parser.add_argument("-S", type=str, help=HELP_S)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print jobs that would be resubmitted without submitting or rewriting the JSON.",
    )
    return parser.parse_args()


def get_sacct_status_map(start_time=None):
    command = ["sacct", "-u", "$USER"]
    if start_time:
        command.extend(["-S", start_time])
    command.extend(
        ["--format=JobIDRaw,State,ExitCode", "--parsable2", "--noheader"]
    )

    result = run(" ".join(command), shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip() or "Failed to query sacct.", file=sys.stderr)
        sys.exit(result.returncode)

    status_map = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue

        job_id, state, exit_code = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not job_id or "." in job_id:
            continue

        status_map[job_id] = (state, exit_code)

    return status_map


def is_active_state(state):
    return state in ACTIVE_STATES


def is_completed_state(state, exit_code):
    return state == "COMPLETED" and exit_code == "0:0"


def resubmit_job(sample_name, info):
    command = info["command"]
    result = run(command, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        return None

    match = re.search(r"\d+", result.stdout)
    if match is None:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        print(
            f"{sample_name}: could not extract a new job id from sbatch output.",
            file=sys.stderr,
        )
        return None

    return match.group()


def main():
    args = parse_args()

    print("INFO:    Argument -S:", args.S)
    print("INFO:    Job JSON:", args.jobjson)
    print("INFO:    Dry run:", args.dry_run)

    with open(args.jobjson) as handle:
        jobs = json.load(handle)

    status_map = get_sacct_status_map(args.S)
    updated = False

    for sample_name, info in jobs.items():
        job_id = str(info["jobid"])
        status_info = status_map.get(job_id)

        if status_info is None:
            print(
                f"{sample_name}: job {job_id} not found in sacct output. Keeping as is."
            )
            continue

        state, exit_code = status_info

        if is_completed_state(state, exit_code):
            continue

        if is_active_state(state):
            print(f"{sample_name}: still active ({state})")
            continue

        if args.dry_run:
            print(
                f"{sample_name}: would resubmit because status is {state} ({exit_code})"
            )
            continue

        print(f"{sample_name}: resubmitting because status is {state} ({exit_code})")
        new_job_id = resubmit_job(sample_name, info)
        if new_job_id is None:
            continue

        jobs[sample_name]["jobid"] = new_job_id
        updated = True

    if args.dry_run:
        print("\nDry run only. JSON was not modified.")
    elif updated:
        print(f"\nRewriting {args.jobjson}...")
        print("-" * 80)
        print()
        with open(args.jobjson, "w") as handle:
            json.dump(jobs, handle, indent=2)
    else:
        print("\nNo job IDs were updated.")


if __name__ == "__main__":
    main()
