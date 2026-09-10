from subprocess import run, PIPE
import re
import json
import os
import time

start_time = time.time()

from pathlib import Path
import sys
PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "testing"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))


# Get sample names from JSON file
# -----------------------------------
JSON_PATH = 'jsons/new/scratch_central_C1N2_Run3.json' # JSON_PATH = 'jsons/new/scratch_data24.json'
with open(JSON_PATH) as f:
    x = json.load(f)
samples = x['CustomNanoAOD']['dir'].keys()


PREDICT_SCRIPT = os.path.join(PROJECT_DIR, 'testing/vtxFramework_predict24.py')
MODEL_PATH     = '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_PART-1111best_valloss_epoch.pt'
INPUT_BASEDIR  = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/c1n2_run3' # '/scratch-cbe/users/alikaan.gueven/ML_KAAN/20260112/data_24'
job_dict = {}


for sample in samples:
    
    year = str(sample.split('_')[-1]) # if sample != 'isomu2017f': continue  # Skip everything except this sample
    if year != '2024': continue

    print("Starting sample: ", sample)
    INPUT_DIR = os.path.join(INPUT_BASEDIR, sample) # e.g. sample='zjetstonunuht0800_2017'

    command = f'python3 -u {PREDICT_SCRIPT} {INPUT_DIR} {MODEL_PATH}'
    result = run(command, shell=True, capture_output = True, text = True)
    # job_id = re.search("\d+", result.stdout).group()    # Get the number with '\d+'
    # info_dict = {'command': f'sbatch {command}',        # Save command [important for resubmitting]
    #              'jobid':   job_id}                     # Save job_id  [identify the status with sacct]
    # job_dict[sample] = info_dict                        # Add to dict
    print('stdout...')
    print('-'*80)
    print(result.stdout[:-1])
    print('stderr...')
    print('-'*80)
    print(result.stderr[:-1])

# study = 'MC_RunIISummer20UL18'
study='MLNano'
out_json_path = os.path.join(INPUT_BASEDIR, f'job_ids_{study}.json')
print(f"\nWriting to {out_json_path}...\n")
with open(out_json_path, 'w') as f:
    json.dump(job_dict, f)

print('\nFinished. Exiting...')
end_time = time.time()
print("Elapsed time: ", (end_time - start_time), " seconds")
print("Elapsed time: ", (end_time - start_time)/60, " minutes")
print("Elapsed time: ", (end_time - start_time)/3600, " hours")
# End of file