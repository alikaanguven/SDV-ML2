import json
import re
/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL18_20241003.json
json_in  = "/eos/vbc/experiments/cms/store/user/aguven/MC_RunIISummer20UL18_20241003.json"
json_out = "jsons/CustomNanoAOD_ML_GNN_test_sorted.json"

with open(json_in) as f:
    data = json.load(f)

re_sig = re.compile(r"(.+)_M(\d+)_(\d+)_ct([0-9p]+)_(\d{4})$")
re_ht  = re.compile(r"(.+?)ht(\d+)_([0-9]{4})$")
re_yr  = re.compile(r"(.+)_([0-9]{4})$")

def keyfunc(s: str):
    # 1) Signal: stopML_M1400_1388_ct2_2018 or ct0p2 etc
    m = re_sig.match(s)
    if m:
        model, M1, M2, ct, year = m.groups()
        ct_val = float(ct.replace("p", "."))
        # group=1 means "signal group"
        return (1, model, int(M1), int(M2), ct_val, int(year), s)

    # 2) HT-binned backgrounds: qcdht0100_2018, wjetstolnuht0400_2018, ...
    m = re_ht.match(s)
    if m:
        prefix, ht, year = m.groups()
        # group=0 means "background group"
        return (0, prefix, int(ht), int(year), s)

    # 3) Other backgrounds with year suffix: ttbar_2018, st_tW_t_2018, ...
    m = re_yr.match(s)
    if m:
        prefix, year = m.groups()
        return (0, prefix, -1, int(year), s)

    # 4) Fallback: anything else
    return (2, s)

d = data["CustomNanoAOD"]["dir"]
data["CustomNanoAOD"]["dir"] = dict(sorted(d.items(), key=lambda kv: keyfunc(kv[0])))

with open(json_out, "w") as f:
    json.dump(data, f, indent=4)

print(f"Wrote sorted json: {json_out}")
