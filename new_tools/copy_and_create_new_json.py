#!/usr/bin/env python3
import json
import shutil
from pathlib import Path

json_in  = "/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1065_IVF/jsons/CustomNanoAOD_ML_IVF_test.json"
json_out = "jsons/clip_IVF_test.json"

out_base = Path("/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/IVF_test")


with open(json_in) as f:
    data = json.load(f)

src_dirs = data["CustomNanoAOD"]["dir"]

out_base.mkdir(parents=True, exist_ok=True)

new_data = {"CustomNanoAOD": {"dir": {}}}

for key, src in src_dirs.items():
    src_path = Path(src)
    dst_path = out_base / key
    dst_path.mkdir(parents=True, exist_ok=True)

    # copy all *.root files from src -> dst (non-recursive)
    for root_file in src_path.glob("*.root"):
        shutil.copy2(root_file, dst_path / root_file.name)

    new_data["CustomNanoAOD"]["dir"][key] = str(dst_path)

with open(json_out, "w") as f:
    json.dump(new_data, f, indent=4)

print(f"Done. Wrote: {json_out}")
