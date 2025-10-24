"""
Usage:       ------
Description: -----
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "training"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_auc_score

from collections import defaultdict



import networks.ParT_ABCDiscoTEC_split as ParT
import user_scripts.preprocess as preprocess
from   user_scripts.branches_to_get import get_branchDict
import user_scripts.val_plots as val_plots
import utils.network_helpers as nh

from utils.vtxLevelDataset import ModifiedUprootIterator
from utils.help_preprocess import probe_shapes
from utils.optimizers.ranger import Ranger

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
# import torch.nn.functional as F
# from torch.profiler import profile, record_function, ProfilerActivity

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


warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
# machine_dependent_defifinitions

hostname = os.uname()[1]
if 'hepgpu' in hostname:
    DATA_READ_BASEPATH  = '/scratch/agueven/ParT/datasets'                               # Change here on HEPGPU!!!
    RUN_SAVE_BASEPATH   = '/scratch/agueven/ParT/runs'                                   # Change here on HEPGPU!!!
    MODEL_SAVE_BASEPATH = '/scratch/agueven/ParT/models'                                 # Change here on HEPGPU!!!
    gpus = [3] # normally 2
elif 'clip' in hostname:
    DATA_READ_BASEPATH  = '/scratch-cbe/users/alikaan.gueven/ML_KAAN'
    RUN_SAVE_BASEPATH   = '/groups/hephy/cms/alikaan.gueven/ParT/runs'
    MODEL_SAVE_BASEPATH = '/groups/hephy/cms/alikaan.gueven/ParT/models'
    gpus = [0]
else:
    raise ValueError('Which machine is this? Seems like this is not clip or hepgpu.')


gpus_str = [str(gpu) for gpu in gpus]
# ------------------------------------------------------------


# json_file = "/groups/hephy/cms/ang.li/MLjson/CustomNanoAOD_MLtraining_20250910.json"
# with open(json_file, "r") as f:
#     data = json.load(f)

# glob_dirs = []
# for key, value in data["CustomNanoAOD_MLtraining_20250910"]["dir"].items():
#     glob_dirs.append(value)

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
minVal   = round(len(tmpSigList)*0.80)
maxVal   = round(len(tmpSigList)*1.00)



trainSigList = tmpSigList[minTrain:maxTrain]
valSigList   = tmpSigList[minVal:maxVal]

extraValList = glob.glob(os.path.join(DATA_READ_BASEPATH, 'ML_validation_exta_bkg/**/*.root'), recursive=True)
nonclBkgList = [val + ':Events' for val in extraValList]


trainDict = {
    'sig': trainSigList,
    'bkg': None
}

valDict = {
    'sig': valSigList,
    'bkg': None
}

nonclDict = {
    'sig': nonclBkgList,
    'bkg': None
}

branchDict = get_branchDict()

shuffle = False
nWorkers = 4


base_step_size = 5000
step_size = base_step_size # * len(gpus)


preprocess_fn = partial(preprocess.transform, branch_dict=branchDict)


prefetch_factor = 16

trainDataset = ModifiedUprootIterator(trainDict,
                                      branchDict,
                                      shuffle=shuffle,
                                      nWorkers=nWorkers,
                                      step_size=step_size)

trainLoader = torch.utils.data.DataLoader(trainDataset, 
                                          num_workers=nWorkers,
                                          prefetch_factor=prefetch_factor,
                                          persistent_workers= True,
                                          collate_fn=preprocess_fn,
                                          drop_last=True, 
                                          pin_memory=True)


valDataset = ModifiedUprootIterator(valDict, 
                                    branchDict,
                                    shuffle=shuffle,
                                    nWorkers=nWorkers,
                                    step_size=step_size)

valLoader = torch.utils.data.DataLoader(valDataset,
                                        num_workers=nWorkers,
                                        prefetch_factor=prefetch_factor,
                                        persistent_workers= True,
                                        collate_fn=preprocess_fn,
                                        pin_memory=True)

nonclDataset = ModifiedUprootIterator(nonclDict,
                                      branchDict,
                                      shuffle=shuffle,
                                      nWorkers=nWorkers,
                                      step_size=step_size)

nonclLoader = torch.utils.data.DataLoader(nonclDataset,
                                          num_workers=nWorkers,
                                          prefetch_factor=prefetch_factor,
                                          persistent_workers= True,
                                          collate_fn=preprocess_fn,
                                          pin_memory=True)



# Training related 
########################################################################

input_shapes = probe_shapes(ModifiedUprootIterator,
                            trainDict,
                            branchDict,
                            preprocess_fn,
                            step_size=step_size)

param = {
    "input_dim":       input_shapes['tk_features'][1],
    "input_svdim":     input_shapes['sv_features'][1],
    "pair_input_dim":  input_shapes['tk_pair_features'][1],
    "embed_dims":      [128, 512, 128],
    "pair_embed_dims": [64, 64, 64],
    "num_classes": 1,
    "for_inference": False,
    "init_lr": 1e-4,
    "class_weights": [1, 1],                # [bkg, sig]
    "init_step_size": step_size,
    "block_params": {'dropout': 0.20, 'attn_dropout': 0.15, 'activation_dropout': 0.15},
    "num_layers": 4,
    "use_amp": False,
    "report_interval": 10,
    "loss_params": {
        'b1': 'random.uniform',
        'b2': 'random.uniform',
        'k': 100.0,
        'eps_closure': 0.1,
        'eps_disco': 0.1,
        'alpha_lr': 1e-5
        },
    "fc_params": [(64, 0.2)]
    }

if (param['loss_params']['b1'].startswith('random')) and (param['loss_params']['b2'].startswith('random')):
    from scipy.stats import rv_continuous
    # Define a linear PDF: p(x) ∝ x over [0, 1]
    class linear_pdf(rv_continuous):
        def _pdf(self, x):
            return (2 * x + 1) / 2 # normalized on [0,1] (since ∫0^1 2x dx = 1)

# Log
########################################################################
use_neptune=True

from shutil import copytree, ignore_patterns

if use_neptune:
    # Set the environment variable for neptune
    api_token = Path.home().joinpath("neptune_api_keys/api_token.txt").read_text().strip()
    os.environ["NEPTUNE_API_TOKEN"] = api_token

    run = neptune.init_run(
        project="alikaan.guven/ParT",
        source_files=[__file__,
                      preprocess.__file__,
                      ParT.__file__,
                      ABCD.__file__,
                      ]
    )


if use_neptune:
    run_savename = "vtx_" + run["sys/id"].fetch()
else:
    run_savename = "vtx" + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')

cp_dest = os.path.join(RUN_SAVE_BASEPATH, run_savename)
copytree(PROJECT_DIR,
        cp_dest,
        ignore=ignore_patterns('*.pyc', 'tmp*', '*.root', '*.pt', '*.png', '*.pdf', '*.ipynb_checkpoints', '__pycache__', '*.ipynb', 'tb*', '.neptune*', 'neptune_key*'))

if use_neptune:
    run["parameters"] = stringify_unsupported(param)


model = ParT.ParticleTransformerDVTagger(input_dim      = param['input_dim'],
                                         input_svdim    = param['input_svdim'],
                                         num_classes    = param['num_classes'],
                                         pair_input_dim = param['pair_input_dim'],
                                         embed_dims     = param['embed_dims'],
                                         for_inference  = param['for_inference'],
                                         block_params   = param['block_params'],
                                         use_amp        = param['use_amp'],
                                         num_layers     = param['num_layers'],
                                         fc_params      = param['fc_params']
                                         )



print('CPU count: ', torch.multiprocessing.cpu_count())
if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs!\n\n")
    model = nn.DataParallel(model, device_ids = gpus)

device = f'cuda:{gpus[0]}'

model.to(device, dtype=torch.float32)
optimizer = Ranger(model.parameters(), lr=param['init_lr'], weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=5, gamma=0.75)
criterion = ABCD.ABCLagrangian(
    param['loss_params']['eps_closure'],
    param['loss_params']['eps_disco'],
    param['loss_params']['k'],
    param['loss_params']['alpha_lr'],
).to(device)


stats = nh.parameter_stats(model)
print(f"Total params        : {stats['total']:,}")
print(f"  Trainable         : {stats['trainable']:,}")
print(f"    • weights       : {stats['trainable_weights']:,}")
print(f"    • biases        : {stats['trainable_biases']:,}")
print(f"  Non-trainable     : {stats['non_trainable']:,}\n")

def train_step(X, batch_num, losses):
    if batch_num == 0:
        print('Started batch processes. [train]')


    optimizer.zero_grad()

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


    # Training related 
    ########################################################################
    output = model(x=tk_features,
                   v=tk_pair_features,
                   x_sv=sv_features,
                   mask=tk_mask)

    # Setting the weights with predetermined class inbalance
    # sample_weights = torch.sum((y==1) * class_weights_tensor,axis=-1)

    logit1 = output['logit1'].squeeze(-1)
    logit2 = output['logit2'].squeeze(-1)

    # print('logit1.shape: ', logit1.shape)
    # print('logit2.shape: ', logit2.shape)
    # print('y.shape: ', y.shape)

    if (param['loss_params']['b1'] == 'random.uniform') and (param['loss_params']['b2'] == 'random.uniform'):
        b1 = np.random.uniform(0.01, 0.99)
        b2 = np.random.uniform(0.01, 0.99)
    elif (param['loss_params']['b1'] == 'random.linear') and (param['loss_params']['b2'] == 'random.linear'):
        linear_dist = linear_pdf(a=0, b=1, name='linear')
        b1 = linear_dist.rvs()
        b2 = linear_dist.rvs()
    else:
        b1 = param['loss_params']['b1']
        b2 = param['loss_params']['b2']

    loss, someLogs = criterion(logit1, logit2, y, b1, b2)
    

    for k, v in someLogs.items():
        losses[k].append(v)
        if batch_num %param['report_interval'] == 0:
            if use_neptune:
                run[f"train/{k}"].append(v)
            else:
                print(f'{k}: {v}')

    if not use_neptune and (batch_num %param['report_interval'] == 0):
        a = torch.cuda.memory_allocated(gpus[0]) / 1e9
        r = torch.cuda.memory_reserved(gpus[0]) / 1e9
        m = torch.cuda.max_memory_allocated(gpus[0]) / 1e9
        print(f"alloc={a:.2f} GB, reserved={r:.2f} GB, max={m:.2f} GB")


    loss.backward()
    optimizer.step()
    criterion.dual_ascent()
    


def validation_step(X, batch_num, losses, p1_bucket, p2_bucket, label_bucket, logName='val'):
    if batch_num == 0:
        print('Started batch processes. [validation]')


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


    # Validation related 
    ########################################################################
    output = model(x=tk_features,
                   v=tk_pair_features,
                   x_sv=sv_features,
                   mask=tk_mask)


    logit1 = output['logit1'].squeeze(-1)
    logit2 = output['logit2'].squeeze(-1)

    if (param['loss_params']['b1'] == 'random.uniform') and (param['loss_params']['b2'] == 'random.uniform'):
        b1 = np.random.uniform(0.01, 0.99)
        b2 = np.random.uniform(0.01, 0.99)
    elif (param['loss_params']['b1'] == 'random.linear') and (param['loss_params']['b2'] == 'random.linear'):
        linear_dist = linear_pdf(a=0, b=1, name='linear')
        b1 = linear_dist.rvs()
        b2 = linear_dist.rvs()
    else:
        b1 = param['loss_params']['b1']
        b2 = param['loss_params']['b2']

    loss, someLogs = criterion(logit1, logit2, y, b1, b2)

    p1 = torch.sigmoid(logit1)
    p2 = torch.sigmoid(logit2)

    p1_bucket.append(p1.detach().cpu())
    p2_bucket.append(p2.detach().cpu())
    label_bucket.append(y.detach().cpu())



    for k, v in someLogs.items():
        losses[k].append(v)
        if batch_num %param['report_interval'] == 0:
            if use_neptune:
                run[f"{logName}/{k}"].append(v)
            else:
                print(f'{k}: {v}')

    if not use_neptune and (batch_num %param['report_interval'] == 0):
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
        m = torch.cuda.max_memory_allocated() / 1e9
        print(f"alloc={a:.2f} GB, reserved={r:.2f} GB, max={m:.2f} GB")




num_epochs = 100


class_weights_tensor = torch.tensor(param['class_weights']).to(device, dtype=torch.float32)
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('Epoch ', epoch)
    print('Starting train...')

    losses   = defaultdict(list)
    
    model.train()

    
    if use_neptune:
        run['parameters/step_size'].append(trainLoader.dataset.step_size)
        run['parameters/lr'].append(scheduler.get_last_lr()[0])
    else:
        print(f"step_size: {trainLoader.dataset.step_size}")
        print(f"lr: {scheduler.get_last_lr()}")
        print(type(scheduler.get_last_lr()[0]))


    for batch_num, X in enumerate(trainLoader):
        train_step(X, batch_num, losses)


    losses_epoch = {}
    for k,v in losses.items():
        losses_epoch[k] = sum(v)/len(v)

    
   
    for k,v in losses_epoch.items():
        if use_neptune:
            run[f'train/{k}_epoch'].append(v if math.isfinite(v) else 999.)
        else:
            print(f'{k} [epoch]: ', v)

    gc.collect() # counter memory leaks at the end of each epoch


    # Validation related 
    ########################################################################
    print('\n'*2)
    print('Entering validation phase...')

    losses   = defaultdict(list)
    p1_bucket, p2_bucket = [], []
    label_bucket  = []


    model.eval()
    with torch.no_grad():
        print("torch.no_grad()")
        for batch_num, X in enumerate(valLoader):
            validation_step(X, batch_num, losses, p1_bucket, p2_bucket, label_bucket)


        losses_epoch = {}
        for k,v in losses.items():
            losses_epoch[k] = sum(v)/len(v)

        
    
        for k,v in losses_epoch.items():
            if use_neptune:
                run[f'val/{k}_epoch'].append(v if math.isfinite(v) else 999.)
            else:
                print(f'{k} [epoch]: ', v)


        ## min loss epoch save
        if losses_epoch['loss'] < best_val_loss:
            suffix = 'best_valloss_epoch.pt'
            best_val_loss = losses_epoch['loss']
            savename = None
            if use_neptune:
                savename = run["sys/id"].fetch() + suffix
            else:
                savename = 'ParT_modified' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + suffix
            # torch.save(model.state_dict(), '/users/alikaan.gueven/ParticleTransformer/PyTorchExercises/models/vtx_' + savename)
            torch.save(model, MODEL_SAVE_BASEPATH + 'vtx_' + savename)
        
        if use_neptune:
            torch.save(model, MODEL_SAVE_BASEPATH + 'vtx_' + run["sys/id"].fetch() + '_epoch_' + str(epoch) + '.pt')
        else:
            torch.save(model, MODEL_SAVE_BASEPATH + 'vtx_' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + '_epoch_' + str(epoch) + '.pt')


        if use_neptune:
            isMatched = torch.cat(label_bucket).to(dtype=torch.bool)
            p1s    = torch.cat(p1_bucket)
            p2s    = torch.cat(p2_bucket)

            val_plots.plot_hist1(p1s, isMatched, "p1_hist", run)
            val_plots.plot_hist1(p2s, isMatched, "p2_hist", run)
            val_plots.plot_hist2(p1s, p2s, isMatched, "p1p2_hist", run)

        # ----------- non-closure     -----------
        if use_neptune:
            losses   = defaultdict(list)
            p1_bucket, p2_bucket = [], []
            label_bucket  = []
            print('Non-closure check')
            for batch_num, X in enumerate(nonclLoader):
                validation_step(X, batch_num, losses, p1_bucket, p2_bucket, label_bucket)
            
            isMatched = torch.cat(label_bucket).to(dtype=torch.bool)
            p1s    = torch.cat(p1_bucket)
            p2s    = torch.cat(p2_bucket)

            x_thresh = np.linspace(0.5, 1.0, 21)
            y_thresh = np.linspace(0.5, 1.0, 21)
            val_plots.plot_only_nonclosure(p1s, p2s, isMatched, x_thresh, y_thresh, "p1p2_hist", run)
            

    scheduler.step()
    gc.collect() # counter memory leaks at the end of each epoch

    # Check if the peak memory increases after each epoch!!
    # if torch.cuda.device_count():
    #     torch.cuda.empty_cache()      # ← releases the cached blocks
    
if use_neptune:
    run.stop()

