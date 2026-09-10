# import os
# import json
# import shutil

# # JSON_FILE = '/eos/vbc/experiments/cms/store/user/aguven/MC_Run3Summer24_20260330.json'
# # OUTDIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/IVF_2024_MC'


# # JSON_FILE = '/users/alikaan.gueven/AOD_to_nanoAOD/Plotter_run3/CMSSW_15_0_5/src/SoftDisplacedVertices/Samples/json/MC_Run3Summer23_20250409.json'
# # OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/IVF_2023_MC'

# JSON_FILE = '/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/clip_IVF_data_2023.json'
# OUTDIR    = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/IVF_2023_data_merged'


# with open('/eos/vbc/experiments/cms/store/user/aguven/MC_Run3Summer24_20260330.json', 'r') as JSON:
#     json_dict = json.load(JSON)


# for k,v in json_dict['CustomNanoAOD']['dir'].items():
#     dest = os.path.join(OUTDIR, k)
#     print(k, '→', dest)
#     os.makedirs(dest, exist_ok=True)
#     shutil.copytree(v, dest, dirs_exist_ok=True)
    