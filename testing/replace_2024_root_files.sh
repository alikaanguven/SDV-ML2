#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/c1n2_run3_tmp_copy"
TARGET_DIR="/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/c1n2_run3"
MODE="${1:-}"
CONFIRM="${2:-}"

if [[ "$MODE" == "--dry-run" && $# -eq 1 ]]; then
    :
elif [[ "$MODE" == "--replace" && ( $# -eq 1 || "$CONFIRM" == "--yes" ) ]]; then
    :
else
    echo "Usage: $0 --dry-run | --replace [--yes]"
    exit 2
fi

if [[ ! -d "$SOURCE_DIR" || ! -d "$TARGET_DIR" ]]; then
    echo "Source or target directory does not exist."
    exit 1
fi

source_files=()
target_files=()
missing_targets=0

while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"$SOURCE_DIR"/}"
    target_file="$TARGET_DIR/$relative_path"

    if [[ ! -f "$target_file" ]]; then
        echo "Missing target: $target_file"
        ((missing_targets += 1))
        continue
    fi

    source_files+=("$source_file")
    target_files+=("$target_file")
done < <(find "$SOURCE_DIR" -type f -path '*/C1N2_*_2024/*' -name '*.root' -print0)

if (( ${#source_files[@]} == 0 )); then
    echo "No matching 2024 ROOT files found."
    exit 1
fi

for i in "${!source_files[@]}"; do
    printf '%s -> %s\n' "${source_files[$i]}" "${target_files[$i]}"
done

echo "Matching files: ${#source_files[@]}"
echo "Missing targets: $missing_targets"

if [[ "$MODE" == "--dry-run" ]]; then
    exit 0
fi

if (( missing_targets > 0 )); then
    echo "Aborted because some corresponding target files are missing."
    exit 1
fi

if [[ "$CONFIRM" != "--yes" ]]; then
    if [[ ! -t 0 ]]; then
        echo "No interactive input available. Use --replace --yes with nohup."
        exit 2
    fi

    read -r -p "Type REPLACE to overwrite these ROOT files: " confirmation
    if [[ "$confirmation" != "REPLACE" ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

for i in "${!source_files[@]}"; do
    cp --preserve=mode,timestamps -- "${source_files[$i]}" "${target_files[$i]}"
done

echo "Replaced ${#source_files[@]} ROOT files."
