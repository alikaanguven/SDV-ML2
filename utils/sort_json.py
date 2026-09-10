#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def sort_json(value):
    if isinstance(value, dict):
        return {key: sort_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_json(item) for item in value]
    return value


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <json_file>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if input_path.suffix.lower() != ".json":
        print("Input file must be a .json file", file=sys.stderr)
        sys.exit(1)

    output_path = input_path.with_name(f"{input_path.stem}_sorted{input_path.suffix}")

    with input_path.open() as handle:
        data = json.load(handle)

    sorted_data = sort_json(data)

    with output_path.open("w") as handle:
        json.dump(sorted_data, handle, indent=4)
        handle.write("\n")

    print(output_path)


if __name__ == "__main__":
    main()
