import json
import glob
import random

json_file = "/groups/hephy/cms/ang.li/MLjson/CustomNanoAOD_MLtraining_20250910.json"

with open(json_file, "r") as f:
    data = json.load(f)

glob_dirs = []

for key, value in data["CustomNanoAOD_MLtraining_20250910"]["dir"].items():
    glob_dirs.append(value)


tmpSigList = []

for sample_dir in glob_dirs:
    tmpSigList.extend(glob.glob(f'{sample_dir}/*.root', recursive=True))

tmpSigList.sort()            # make deterministic order
random.seed(42)              # set reproducible seed
random.shuffle(tmpSigList)   # shuffle in reproducible way

print(tmpSigList)