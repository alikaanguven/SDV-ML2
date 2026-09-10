#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def directory_map(input_dir: Path) -> dict[str, str]:
    return {
        path.name: str(path.resolve())
        for path in sorted(input_dir.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a JSON file mapping directory names to full paths."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory whose immediate subdirectories should be listed.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("CustomMiniAOD.json"),
        help="Output JSON path. Default: CustomMiniAOD.json",
    )
    parser.add_argument(
        "--top-key",
        default="CustomMiniAOD",
        help="Top-level JSON key. Default: CustomMiniAOD",
    )
    parser.add_argument(
        "--dir-key",
        default="dir",
        help="Directory mapping key under the top-level object. Default: dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    data = {args.top_key: {args.dir_key: directory_map(input_dir)}}
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(data[args.top_key][args.dir_key])} directories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
