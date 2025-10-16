"""
Prediction script

"""

import torch
import torch.nn.functional as F
torch.set_printoptions(precision=15)

import awkward as ak
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from functools import partial

import argparse
import gc
import glob
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "testing"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))




import user_scripts.preprocess as preprocess
from   user_scripts.branches_to_get import get_branchDict
from utils.vtxLevelDataset import ModifiedUprootIterator




print('CPU count: ', torch.multiprocessing.cpu_count())

parser = argparse.ArgumentParser("ParT predict")
parser.add_argument("inputdir",  help="Give us the input directory my precious.",  type=str)
parser.add_argument("model_path", help="Give us the model name my precious.",      type=str)
parser.add_argument("-o", "--outputdir", default=None, help="Give us the output directory my precious. Defaults to inputdir.")


args = parser.parse_args()
if args.outputdir is None: args.outputdir = args.inputdir


INPUTDIR   = args.inputdir
MODEL_PATH = args.model_path # 'vtx_PART-338_epoch_6'
MODEL_NAME = os.path.basename(MODEL_PATH).strip('.pt')
OUTPUT_PARQUET_PATH1  = os.path.join(args.outputdir,      f'{MODEL_NAME}_p1.parquet')
OUTPUT_PARQUET_PATH2  = os.path.join(args.outputdir,      f'{MODEL_NAME}_p2.parquet')



files = glob.glob(f'{INPUTDIR}/**/*.root', recursive=True)
testList = files


branchDict = get_branchDict()

preprocess_fn = partial(preprocess.transform, branch_dict=branchDict)


shuffle = False
nWorkers = 1
step_size = 20000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



model = torch.load(MODEL_PATH, map_location=torch.device(device))

if isinstance(model, torch.nn.DataParallel):
    model = model.module

model.to(device)
model.eval()



save_dict1  = {}
save_dict2 = {}


print("testList: ", testList)


with torch.no_grad():

    for file in testList:

        p1_bucket   = []
        p2_bucket   = []
        print(f'Starting file {file}...')
        testDict = {
            'sig': [file + ':Events'],
            'bkg': []
        }

        prefetch_factor = 16

        testDataset = ModifiedUprootIterator(testDict, 
                                             branchDict,
                                             shuffle=False,
                                             nWorkers=nWorkers,
                                             step_size=step_size)
        
        testLoader = torch.utils.data.DataLoader(testDataset,
                                                 num_workers=nWorkers,
                                                 prefetch_factor=prefetch_factor,
                                                 persistent_workers= True,
                                                 collate_fn=preprocess_fn,
                                                 pin_memory=True)
        
        for batch_num, X in enumerate(testLoader):
            if batch_num == 0:
                print('Started batch processes. [prediction]')


            tk_pair_features = X["tk_pair_features"]
            tk_features      = X["tk_features"]
            tk_mask          = X["tk_mask"]
            sv_features      = X["sv_features"]
            
            y =  (X['label'].squeeze(-1) > 1).long()
            

            tk_pair_features = tk_pair_features.to(device, dtype=torch.float32, non_blocking=True)
            tk_features      = tk_features.to(device,      dtype=torch.float32, non_blocking=True)
            tk_mask          = tk_mask.to(device,          dtype=torch.bool,  non_blocking=True)
            sv_features      = sv_features.to(device,      dtype=torch.float32, non_blocking=True)
            y                = y.to(device,                dtype=torch.float32, non_blocking=True)       


            output = model(x=tk_features,
                           v=tk_pair_features,
                           x_sv=sv_features,
                           mask=tk_mask)
            
            logit1 = output['logit1'].squeeze(-1)
            logit2 = output['logit2'].squeeze(-1)

            p1 = torch.sigmoid(logit1)
            p2 = torch.sigmoid(logit2)

            p1_bucket.append(p1.detach().cpu())
            p2_bucket.append(p2.detach().cpu())
            
            
        
        p1s_np = torch.cat(p1_bucket).numpy()
        p2s_np = torch.cat(p2_bucket).numpy()
        
        save_dict1[file] = np.ascontiguousarray(p1s_np)
        save_dict2[file] = np.ascontiguousarray(p2s_np)
        
        gc.collect() # counter memory leaks at the end of each epoch

        
record1 = ak.Record(save_dict1)
record2 = ak.Record(save_dict2)
ak.to_parquet(record1, OUTPUT_PARQUET_PATH1)
ak.to_parquet(record2, OUTPUT_PARQUET_PATH2)

