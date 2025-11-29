import os
import json
import shutil

with open('/users/ang.li/public/SoftDV/CMSSW_13_3_0/src/SoftDisplacedVertices/Samples/json/CustomNanoAOD_GNNAVRIVF_new2.json', 'r') as JSON:
    json_dict = json.load(JSON)

# /scratch-cbe/users/alikaan.gueven/ML_KAAN/


for k,v in json_dict['CustomNanoAOD']['dir'].items():
    dest = f'/scratch-cbe/users/alikaan.gueven/ML_KAAN/CustomNanoAOD_GNNAVRIVF_new2/{k}'
    print(k, '→', dest)
    os.makedirs(dest, exist_ok=True)
    shutil.copytree(v, dest, dirs_exist_ok=True)
    