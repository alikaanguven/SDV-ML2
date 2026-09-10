import os
import json
import subprocess

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/MC_RunIISummer20UL17_20241003.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/MC_17'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/MC_RunIISummer20UL18_20241003.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/MC_18'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/MC_Run3Summer22_20250621.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/MC_22'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/MC_Run3Summer23_20250409.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/MC_23'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/MC_Run3Summer24_20260330.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/MC_24'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_MET2017.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_17'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_production_20240326.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_18'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/scratch_CustomNanoAOD_v3_centralprod.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/sig_17-18_old_centralprod'

JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_IsoMu2017.json'
OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/muon17'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_Run3_jetmet_official_22.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_22'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_Run3_jetmet_official_23.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_23'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/Data_Run3_jetmet_official_24.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_24'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/CustomNanoAOD_ML_IVF_sorted.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/sig_18'


with open(JSON_FILE, 'r') as JSON:
    json_dict = json.load(JSON)


os.makedirs(OUTDIR, exist_ok=True)
for k,v in json_dict['CustomNanoAOD']['dir'].items():
    dest = os.path.join(OUTDIR, k)
    os.makedirs(dest, exist_ok=True)
    print(k, '→', dest)
    try:
        subprocess.run(['xrdcp', '--recursive', v, dest], check=True)
    except Exception as e:
        print(e)
    # subprocess.run(['xrdcp', '--force', '--recursive', v, dest], check=True)
    
