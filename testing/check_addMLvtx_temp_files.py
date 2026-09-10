#!/usr/bin/env python3

import argparse
import os
import sys

import ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check addMLvtx temporary ROOT files and optionally delete them."
    )
    parser.add_argument(
        "directory",
        help="Directory to scan recursively for *_new.root and *_cleaned.root files.",
    )
    parser.add_argument(
        "--delete-new",
        action="store_true",
        help="Delete files ending with *_new.root after checking them.",
    )
    parser.add_argument(
        "--delete-cleaned",
        action="store_true",
        help="Delete files ending with *_cleaned.root after checking them.",
    )
    return parser.parse_args()


def classify_file(path):
    if path.endswith("_new.root"):
        return "new"
    if path.endswith("_cleaned.root"):
        return "cleaned"
    return None


def check_root_file(path):
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file:
        return False, "could not open"

    try:
        if root_file.IsZombie():
            return False, "zombie"

        tree = root_file.Get("Events")
        if not tree:
            return False, "missing Events tree"

        _ = tree.GetEntries()
        return True, "readable"
    except Exception as exc:
        return False, f"error while reading: {exc}"
    finally:
        root_file.Close()


def should_delete(file_kind, args):
    return (file_kind == "new" and args.delete_new) or (
        file_kind == "cleaned" and args.delete_cleaned
    )


def main():
    args = parse_args()
    directory = os.path.abspath(args.directory)

    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}", file=sys.stderr)
        sys.exit(1)

    matches = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_kind = classify_file(full_path)
            if file_kind is not None:
                matches.append((file_kind, full_path))

    matches.sort(key=lambda item: item[1])

    if not matches:
        print("No *_new.root or *_cleaned.root files found.")
        return

    deleted = 0
    for file_kind, path in matches:
        ok, message = check_root_file(path)
        status = "OK" if ok else "BROKEN"
        print(f"[{file_kind.upper():7}] [{status:6}] {path} :: {message}")

        if should_delete(file_kind, args):
            os.remove(path)
            deleted += 1
            print(f"Deleted: {path}")

    n_new = sum(1 for file_kind, _ in matches if file_kind == "new")
    n_cleaned = sum(1 for file_kind, _ in matches if file_kind == "cleaned")
    print()
    print(f"Found {len(matches)} temporary files in total.")
    print(f"  new: {n_new}")
    print(f"  cleaned: {n_cleaned}")
    print(f"Deleted: {deleted}")


if __name__ == "__main__":
    main()
