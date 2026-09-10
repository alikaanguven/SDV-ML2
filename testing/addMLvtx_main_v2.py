import os
import re
import json
from subprocess import run
import pandas as pd
import utils.root_helpers as root_helpers



INPUTBASE_DIR = "/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/muon17"
MODEL_NAME = 'vtx_PART-1111best_valloss_epoch'


run2_jsons = {
    "data":  ["/groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/jsons/new/scratch_mu17.json"],
    }

job_dict = {}
files_per_job = 4

for tag, jsonfile_list in run2_jsons.items():
    tier = "CustomNanoAOD"

    for jsonfile in jsonfile_list:
        with open(jsonfile,'r') as f:
            json_dict = json.load(f)
        
        for sample in json_dict[tier]['dir'].keys():
            # if sample != 'jetmet_2022e': continue  # Skip everything except this sample
            INPUT_DIR = os.path.join(INPUTBASE_DIR, sample)
            PRED_PATH = os.path.join(INPUT_DIR, f'{MODEL_NAME}.parquet')
            df = pd.read_parquet(PRED_PATH)

            files = df.columns
            len_files = len(files)
            chunks = [files[i:i+files_per_job] for i in range(0, len(files), files_per_job)]
            for i, chunk in enumerate(chunks):
                INPUT_FILES = ",".join(chunk)
                
                root_helpers.remove_new_root_files(INPUT_DIR)
                command = f'sbatch slurm_scripts/to_cpu1.sh "python3 -u /groups/hephy/cms/alikaan.gueven/ParT/runs/vtx_PART-1111/testing/addMLvtx_subprocess_v2.py {INPUT_FILES} {PRED_PATH} --skip_existing_branch"' 
                result = run(command, shell=True, capture_output = True, text = True)
                
                job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
                info_dict = {'command': f'{command}',               # Save command [important for resubmitting]
                             'jobid':   job_id}                     # Save job_id  [identify the status with sacct]
                job_dict[sample + f'_{i}'] = info_dict              # Add to dict
                print(result.stdout[:-1])

out_json_dir = os.path.join(INPUTBASE_DIR, 'jobs_fillnano')
os.makedirs(out_json_dir, exist_ok=True)
out_json_path = os.path.join(out_json_dir, f'{MODEL_NAME}.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f)

print('\nFinished. Exiting...')
