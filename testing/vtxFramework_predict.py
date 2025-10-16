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
OUTPUT_PARQUET_PATH  = os.path.join(args.outputdir,      f'{MODEL_NAME}.parquet')
OUTPUT_PARQUET_PATH2 = os.path.join(args.outputdir,      f'{MODEL_NAME}_logitgap.parquet')



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



save_dict  = {}
save_dict2 = {}


print("testList: ", testList)


with torch.no_grad():

    for file in testList:
        print("Starting file: ", file)
        logit_bucket  = []
        prob_bucket   = []
        delta_bucket  = []
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
            
            y = F.one_hot( (X['label'][:,0] > 1).long(), num_classes=2 )
            

            tk_pair_features = tk_pair_features.to(device, dtype=float, non_blocking=True)
            tk_features      = tk_features.to(device,      dtype=float, non_blocking=True)
            tk_mask          = tk_mask.to(device,          dtype=float, non_blocking=True)
            sv_features      = sv_features.to(device,      dtype=float, non_blocking=True)
            y                = y.to(device,                dtype=float, non_blocking=True)       


            
            output = model(x=tk_features,
                           v=tk_pair_features,
                           x_sv=sv_features,
                           mask=tk_mask)
            
            logit_bucket.append(output)                     # https://arxiv.org/pdf/1503.02531
            output_softmax = torch.softmax(output, dim=1)   # multiply output by a value if you want temperature sharpening
            prob_bucket.append(output_softmax)

            logit_gap = output[:,1] - output[:,0]
            delta_bucket.append(logit_gap * 1000.)

            
        
        
        probs_np = torch.cat(prob_bucket)[:,1].cpu().numpy()
        deltas_np = torch.cat(delta_bucket).cpu().numpy()
        save_dict[file]  = np.ascontiguousarray(probs_np)
        save_dict2[file] = np.ascontiguousarray(deltas_np)
        
        gc.collect() # counter memory leaks at the end of each epoch

        
record  = ak.Record(save_dict)
record2 = ak.Record(save_dict2)
ak.to_parquet(record, OUTPUT_PARQUET_PATH)
ak.to_parquet(record2, OUTPUT_PARQUET_PATH2)

