#!/usr/bin/env python3

import re
import sys
from pathlib import Path


KEYWORDS = [
    "error",
    "traceback",
    "exception",
    "segmentation",
    "fault",
    "failed",
    "failure",
    "fatal",
    "abort",
    "aborted",
    "killed",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <log_file>", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 2

    pattern = re.compile("|".join(re.escape(word) for word in KEYWORDS), re.IGNORECASE)
    matches = []

    with log_path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if pattern.search(line):
                matches.append((line_number, line.rstrip("\n")))

    if not matches:
        print(f"No error-like keywords found in {log_path}")
        return 0

    print(f"Found {len(matches)} suspicious line(s) in {log_path}:")
    for line_number, text in matches:
        print(f"{line_number}: {text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
