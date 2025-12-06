import sys
from pathlib import Path

PROJECT_DIR = Path('.').resolve().parent
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score

from collections import defaultdict



import networks.ParT_ABCDiscoTEC_split as ParT
import user_scripts.preprocess as preprocess
from   user_scripts.branches_to_get import get_branchDict
import user_scripts.val_plots2 as val_plots
import utils.network_helpers as nh

from utils.vtxLevelDataset import ModifiedUprootIterator
from utils.help_preprocess import probe_shapes
from utils.optimizers.ranger import Ranger

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import user_scripts.ABCDiscoTEC_loss as ABCD

import neptune
from neptune.utils import stringify_unsupported

import datetime
from functools import partial
import glob
import gc
import math
import warnings
import os
import random
import json
import copy


DATA_READ_BASEPATH  = '/scratch-cbe/users/alikaan.gueven/ML_KAAN'
RUN_SAVE_BASEPATH   = '/groups/hephy/cms/alikaan.gueven/ParT/runs'
MODEL_SAVE_BASEPATH = '/groups/hephy/cms/alikaan.gueven/ParT/models'

glob_dirs = [os.path.join(DATA_READ_BASEPATH, 'CustomNanoAOD_MLtraining_20250910_mixed')]

tmpSigList = []
for sample_dir in glob_dirs:
    tmpSigList.extend(glob.glob(f'{sample_dir}/**/*.root', recursive=True))

tmpSigList.sort()            # make deterministic order
random.seed(42)              # set reproducible seed
random.shuffle(tmpSigList)   # shuffle in reproducible way

tmpSigList = [sig + ':Events' for sig in tmpSigList]


minTrain = round(len(tmpSigList)*0.00)
maxTrain = round(len(tmpSigList)*0.80)

trainSigList = tmpSigList[minTrain:maxTrain]


trainDict = {
    'sig': trainSigList,
    'bkg': None
}


branchDict_dataset    = get_branchDict()

shuffle = False
nWorkers = 1
step_size = 4000

trainDataset = ModifiedUprootIterator(trainDict,
                                      branchDict_dataset,
                                      shuffle=shuffle,
                                      nWorkers=nWorkers,
                                      step_size=step_size)

# Create the iterator first, THEN call next
iterator = iter(trainDataset) 
X = next(iterator)
X['SDVTrack_pt']

prefetch_factor = 16

branchDict_dataloader = copy.deepcopy(branchDict_dataset)
branchDict_dataloader['ev'].append('event_idx')

preprocess_fn = partial(preprocess.transform, branch_dict=branchDict_dataloader)


trainLoader = torch.utils.data.DataLoader(trainDataset, 
                                          num_workers=nWorkers,
                                          prefetch_factor=prefetch_factor,
                                          persistent_workers= True,
                                          collate_fn=preprocess_fn,
                                          drop_last=True, 
                                          pin_memory=True)
summ = 0
for batch_num, X in enumerate(trainLoader):
    df = ABCD.select_leading_vertices(X['sv_features'][:,0,0],
                                      X['sv_features'][:,1,0],
                                      X['event_idx'],
                                      X['label'])
    batch_count = df[-1].shape[0] 
    summ += batch_count
    print(summ)
    print(df[-1].shape, df[0][:5])
