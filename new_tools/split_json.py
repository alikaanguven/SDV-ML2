import json
import re
import copy

json_in = "jsons/CustomNanoAOD_ML_IVF.json"
train_out = "jsons/CustomNanoAOD_ML_IVF_train.json"
test_out  = "jsons/CustomNanoAOD_ML_IVF_test.json"

with open(json_in) as f:
    data = json.load(f)

test_keys = {
    "C1N2ML_M500_485_ct2_2018",
    "C1N2ML_M500_485_ct20_2018",
    "C1N2ML_M500_485_ct200_2018",
    "stopML_M1000_975_ct0p2_2018",
    "stopML_M1000_980_ct2_2018",
    "stopML_M1000_985_ct20_2018",
    "stopML_M1000_988_ct200_2018",
}

def keyfunc(s: str):
    # stopML_M1400_1388_ct2_2018 or ct0p2 etc
    m = re.search(r"(.+)_M(\d+)_(\d+)_ct([0-9p]+)_(\d+)$", s)
    if not m:
        return (s,)  # fallback: keep non-matching keys sortable but harmless
    model, M1, M2, ct, year = m.groups()
    ct = float(ct.replace("p", "."))
    return (model, int(M1), int(M2), ct, int(year))

d = data["CustomNanoAOD"]["dir"]

train_dir = {k: v for k, v in d.items() if k not in test_keys}
test_dir  = {k: v for k, v in d.items() if k in test_keys}

train_dir = dict(sorted(train_dir.items(), key=lambda kv: keyfunc(kv[0])))
test_dir  = dict(sorted(test_dir.items(),  key=lambda kv: keyfunc(kv[0])))

train_data = copy.deepcopy(data)
test_data  = copy.deepcopy(data)
train_data["CustomNanoAOD"]["dir"] = train_dir
test_data["CustomNanoAOD"]["dir"]  = test_dir

with open(train_out, "w") as f:
    json.dump(train_data, f, indent=4)

with open(test_out, "w") as f:
    json.dump(test_data, f, indent=4)
