import glob
import ROOT
import json
import os

ROOT.EnableImplicitMT()  # optional: multithread


json_in =  "jsons/CustomNanoAOD_ML_IVF_train.json"
with open(json_in) as f:
    data = json.load(f)
d = data["CustomNanoAOD"]["dir"]
files = []
for sample in d:
    sample_dir = d[sample]
    sample_files = glob.glob(os.path.join(sample_dir, "**/*.root"), recursive=True)
    files.extend(sample_files)

df = ROOT.RDataFrame("Events", files)       # tree name often "Events"

df2 = (df
    .Define("nSigVtx", "ROOT::VecOps::Sum(SDVSecVtx_matchedLLPnDau_bydau > 1)")
    .Define("nBkgVtx", "ROOT::VecOps::Sum(SDVSecVtx_matchedLLPnDau_bydau == 0)")
)

tot_sig = df2.Sum("nSigVtx").GetValue()
tot_bkg = df2.Sum("nBkgVtx").GetValue()
tot_evt = df2.Count().GetValue()

print("events:", int(tot_evt))
print("signal vertices:", int(tot_sig))
print("background vertices:", int(tot_bkg))
print("total vertices (sig+bkg):", int(tot_sig + tot_bkg))
print("sig/bkg ratio:", tot_sig / tot_bkg if tot_bkg > 0 else "inf")