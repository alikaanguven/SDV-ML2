"""
Usage:       ------
Description: -----
"""
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]  # parent of "training"
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score



import networks.ParT_K as ParT
import user_scripts.preprocess as preprocess
from   user_scripts.branches_to_get import get_branchDict
import utils.network_helpers as nh

from utils.vtxLevelDataset import ModifiedUprootIterator
from utils.help_preprocess import probe_shapes
from utils.optimizers.ranger import Ranger

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

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




# # MLDATADIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/run2/'
# MLDATADIR = '/scratch-cbe/users/alikaan.gueven/ML_KAAN/Ang_GNN_signal_mixed'
# 
# # tmpSigList = glob.glob(f'{MLDATADIR}/stop*/**/*.root', recursive=True)
# tmpSigList = glob.glob(f'{MLDATADIR}/*.root', recursive=True)




json_file = "/groups/hephy/cms/ang.li/MLjson/CustomNanoAOD_MLtraining_20250910.json"
with open(json_file, "r") as f:
    data = json.load(f)

glob_dirs = []
for key, value in data["CustomNanoAOD_MLtraining_20250910"]["dir"].items():
    glob_dirs.append(value)

tmpSigList = []
for sample_dir in glob_dirs:
    tmpSigList.extend(glob.glob(f'{sample_dir}/*.root', recursive=True))

tmpSigList.sort()            # make deterministic order
random.seed(42)              # set reproducible seed
random.shuffle(tmpSigList)   # shuffle in reproducible way

tmpSigList = [sig + ':Events' for sig in tmpSigList]

maxTrain = round(len(tmpSigList)*0.70)
# maxTrain = round(len(tmpSigList)*0.10)

minVal =   round(len(tmpSigList)*0.70)
maxVal   = round(len(tmpSigList)*1.00)



trainSigList = tmpSigList[:maxTrain]
valSigList   = tmpSigList[minVal:maxVal]

# trainBkgList = glob.glob(f'{MLDATADIR}/training_set/bkg_mix*.root')
# valBkgList = glob.glob(f'{MLDATADIR}/val_set/bkg_mix*.root')
# 
# trainBkgList = glob.glob('/scratch-cbe/users/alikaan.gueven/ML_KAAN/train/training_set/bkg_mix*.root')
# valBkgList = glob.glob('/scratch-cbe/users/alikaan.gueven/ML_KAAN/train/val_set/bkg_mix*.root')
# 
# 
# trainBkgList = [elm + ':Events' for elm in trainBkgList]
# valBkgList = [elm + ':Events' for elm in valBkgList]


# trainDict = {
#     'sig': trainSigList,
#     'bkg': trainBkgList
# }
# 
# valDict = {
#     'sig': valSigList,
#     'bkg': valBkgList
# }

trainDict = {
    'sig': trainSigList,
    'bkg': None
}

valDict = {
    'sig': valSigList,
    'bkg': None
}

branchDict = get_branchDict()

shuffle = False
nWorkers = 6
base_step_size = 1000
if torch.cuda.device_count():
    step_size = base_step_size * torch.cuda.device_count()
else:
    step_size = base_step_size

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
                                    step_size=step_size*5)

valLoader = torch.utils.data.DataLoader(valDataset,
                                        num_workers=nWorkers,
                                        prefetch_factor=prefetch_factor,
                                        persistent_workers= True,
                                        collate_fn=preprocess_fn,
                                        pin_memory=True)




# Training related 
########################################################################

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

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
    "num_classes": 2,
    "for_inference": False,
    "init_lr": 8e-4,
    "class_weights": [1, 1],                # [bkg, sig]
    "init_step_size": step_size,
    "block_params": {'dropout': 0.20, 'attn_dropout': 0.15, 'activation_dropout': 0.15},
    "num_layers": 4,
    "use_amp": False
}

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
        source_files=[os.path.join(PROJECT_DIR, __file__),
                      os.path.join(PROJECT_DIR, preprocess.__file__),
                      os.path.join(PROJECT_DIR, ParT.__file__)]
    )


if use_neptune:
    run_savename = "vtx_" + run["sys/id"].fetch()
else:
    run_savename = "vtx" + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S')

destination = os.path.join('/groups/hephy/cms/alikaan.gueven/ParT/runs', run_savename)
copytree(PROJECT_DIR,
        destination,
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
                                         )

print('CPU count: ', torch.multiprocessing.cpu_count())
if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs!\n\n")
    model = nn.DataParallel(model)


model.to(device, dtype=float)
optimizer = Ranger(model.parameters(), lr=param['init_lr'], weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=4, gamma=0.75)
criterion = nn.CrossEntropyLoss(reduction='none')


stats = nh.parameter_stats(model)
print(f"Total params        : {stats['total']:,}")
print(f"  Trainable         : {stats['trainable']:,}")
print(f"    • weights       : {stats['trainable_weights']:,}")
print(f"    • biases        : {stats['trainable_biases']:,}")
print(f"  Non-trainable     : {stats['non_trainable']:,}\n")

def train_step(X, batch_num, CM_epoch, losses):
    if batch_num == 0:
        print('Started batch processes. [train]')


    optimizer.zero_grad()

    tk_pair_features = X["tk_pair_features"]
    tk_features      = X["tk_features"]
    tk_mask          = X["tk_mask"]
    sv_features      = X["sv_features"]
    
    y = F.one_hot( (X['label'].squeeze(-1) > 1).long(), num_classes=2 )
    

    tk_pair_features = tk_pair_features.to(device, dtype=float, non_blocking=True)
    tk_features      = tk_features.to(device,      dtype=float, non_blocking=True)
    tk_mask          = tk_mask.to(device,          dtype=bool,  non_blocking=True)
    sv_features      = sv_features.to(device,      dtype=float, non_blocking=True)
    y                = y.to(device,                dtype=float, non_blocking=True)       


    # Training related 
    ########################################################################
    output = model(x=tk_features,
                   v=tk_pair_features,
                   x_sv=sv_features,
                   mask=tk_mask)

    # Setting the weights with predetermined class inbalance
    sample_weights = torch.sum((y==1) * class_weights_tensor,axis=-1)

    
    loss = criterion(output, y)
    loss = torch.mean(sample_weights * loss)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    output = torch.softmax(output, dim=1)

    sigThreshold = 0.50
    y_pred01 = (output[:,-1] > sigThreshold).to('cpu', dtype=int)
    y_test01 = y.data[:,-1].to('cpu', dtype=int)
    CM = confusion_matrix(y_test01, y_pred01, labels=[0, 1])
    TN, FP, FN, TP = CM.ravel()

    CM_epoch['TP'] += TP
    CM_epoch['FN'] += FN
    CM_epoch['FP'] += FP
    CM_epoch['TN'] += TN

        
    TPR = TP / (TP+FN) if (TP+FN) != 0 else 0
    PPV = TP / (TP+FP) if (TP+FP) != 0 else 0

    if use_neptune:
        run["train/TPR"].append(TPR if math.isfinite(TPR) else 0)
        run["train/PPV"].append(PPV if math.isfinite(PPV) else 0)

    elif batch_num %10 == 0:
        print('batch_num: ', batch_num)
        print('Class imbalance: ', [round((torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)).item(),2), 1]) # bkg/sig
        print('#'*80)
        print('#'*80)
        print('TPR: ', TPR)
        print('PPV: ', PPV)
        print('CM:')
        print(CM)
        


    acc = (TP+TN) / (TP + FN + FP + TN)
    if use_neptune:
        run["train/accuracy_batch"].append(acc if math.isfinite(acc) else 0)
        run["train/loss_batch"].append(loss.item() if math.isfinite(loss.item()) else 10)
        
    else:
        if batch_num %10 == 0:
            print('Acc:  ', acc.item())
            print('Loss: ', loss.item())

def validation_step(X, batch_num, CM_epoch, losses, output_bucket, label_bucket):
    if batch_num == 0:
        print('Started batch processes. [validation]')


    tk_pair_features = X["tk_pair_features"]
    tk_features      = X["tk_features"]
    tk_mask          = X["tk_mask"]
    sv_features      = X["sv_features"]
    
    y = F.one_hot( (X['label'].squeeze(-1) > 1).long(), num_classes=2 )
    

    tk_pair_features = tk_pair_features.to(device, dtype=float, non_blocking=True)
    tk_features      = tk_features.to(device,      dtype=float, non_blocking=True)
    tk_mask          = tk_mask.to(device,          dtype=bool,  non_blocking=True)
    sv_features      = sv_features.to(device,      dtype=float, non_blocking=True)
    y                = y.to(device,                dtype=float, non_blocking=True)       


    # Validation related 
    ########################################################################
    output = model(x=tk_features,
                   v=tk_pair_features,
                   x_sv=sv_features,
                   mask=tk_mask)

    # Setting the weights with predetermined class inbalance
    sample_weights = torch.sum((y==1) * class_weights_tensor,axis=-1)

    
    loss = criterion(output, y)
    loss = torch.mean(sample_weights * loss)
    losses.append(loss.item())
    output = torch.softmax(output, dim=1)

    output_bucket.append(output)
    label_bucket.append(y.data)

    sigThreshold = 0.50
    y_pred01 = (output[:,-1] > sigThreshold).to('cpu', dtype=int)
    y_test01 = y.data[:,-1].to('cpu', dtype=int)
    CM = confusion_matrix(y_test01, y_pred01, labels=[0, 1])
    TN, FP, FN, TP = CM.ravel()

    CM_epoch['TP'] += TP
    CM_epoch['FN'] += FN
    CM_epoch['FP'] += FP
    CM_epoch['TN'] += TN

        
    TPR = TP / (TP+FN) if (TP+FN) != 0 else 0
    PPV = TP / (TP+FP) if (TP+FP) != 0 else 0

    if use_neptune:
        run["val/TPR"].append(TPR if math.isfinite(TPR) else 0)
        run["val/PPV"].append(PPV if math.isfinite(PPV) else 0)

    elif batch_num %10 == 0:
        print('batch_num: ', batch_num)
        print('Class imbalance: ', [round((torch.sum(y.data[:,-1] == 0) / torch.sum(y.data[:,-1] == 1)).item(),2), 1]) # bkg/sig
        print('#'*80)
        print('#'*80)
        print('TPR: ', TPR)
        print('PPV: ', PPV)
        print('CM:')
        print(CM)
        


    acc = (TP+TN) / (TP + FN + FP + TN)
    if use_neptune:
        run["val/accuracy_batch"].append(acc if math.isfinite(acc) else 0)
        run["val/loss_batch"].append(loss.item() if math.isfinite(loss.item()) else 10)
        
    else:
        if batch_num %10 == 0:
            print('Acc:  ', acc.item())
            print('Loss: ', loss.item())


num_epochs = 200


class_weights_tensor = torch.tensor(param['class_weights']).to(device, dtype=float)
best_val_acc  = 0
best_val_loss = np.inf

for epoch in range(num_epochs):
    print('Epoch ', epoch)
    print('Starting train...')
    CM_epoch = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    losses   = []
    
    model.train()

    
    if use_neptune:
        run['parameters/step_size'].append(trainLoader.dataset.step_size)
        run['parameters/lr'].append(scheduler.get_last_lr()[0])
    else:
        print(f"step_size: {trainLoader.dataset.step_size}")
        print(f"lr: {scheduler.get_last_lr()}")
        print(type(scheduler.get_last_lr()[0]))


    for batch_num, X in enumerate(trainLoader):
        train_step(X, batch_num, CM_epoch, losses)


    acc_epoch  = (CM_epoch['TP'] + CM_epoch['TN']) / (CM_epoch['TP'] + CM_epoch['FN'] + CM_epoch['FP'] + CM_epoch['TN'])
    TPR_epoch  = CM_epoch['TP'] / (CM_epoch['TP'] + CM_epoch['FN'])
    PPV_epoch  = CM_epoch['TP'] / (CM_epoch['TP'] + CM_epoch['FP'])
    loss_epoch = sum(losses)/len(losses)
    
    if use_neptune:
        run["train/accuracy_epoch"].append(acc_epoch if math.isfinite(acc_epoch) else 0)
        run["train/TPR_epoch"].append(TPR_epoch if math.isfinite(TPR_epoch) else 0)
        run["train/PPV_epoch"].append(PPV_epoch if math.isfinite(PPV_epoch) else 0)
        run["train/losses_epoch"].append(loss_epoch if math.isfinite(loss_epoch) else 10)
    else:
        print('Acc  [epoch]: ', acc_epoch)
        print('Loss [epoch]: ', loss_epoch)

    
    # Validation related 
    ########################################################################
    print('\n'*2)
    print('Entering validation phase...')
    CM_epoch = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}
    losses        = []
    output_bucket = []
    label_bucket  = []


    model.eval()
    with torch.no_grad():
        print("torch.no_grad()")
        for batch_num, X in enumerate(valLoader):
            validation_step(X, batch_num, CM_epoch, losses, output_bucket, label_bucket)

        # --- AUC ---------------------------------------------------------------
        # true labels: 1 for signal, 0 for background
        y_true   = torch.cat(label_bucket)[:, 1].cpu().numpy()

        # probability assigned to the positive class
        y_scores = torch.cat(output_bucket)[:, 1].cpu().numpy()

        auc_epoch = roc_auc_score(y_true, y_scores)
        # ----------------------------------------------------------------------

        if use_neptune:
            run["val/auc_epoch"].append(auc_epoch if math.isfinite( auc_epoch) else 0)
        else:
            print(f"AUC  [epoch]: {auc_epoch:.4f}")

            
        acc_epoch  = (CM_epoch['TP'] + CM_epoch['TN']) / (CM_epoch['TP'] + CM_epoch['FN'] + CM_epoch['FP'] + CM_epoch['TN'])
        TPR_epoch  = CM_epoch['TP'] / (CM_epoch['TP'] + CM_epoch['FN'])
        PPV_epoch  = CM_epoch['TP'] / (CM_epoch['TP'] + CM_epoch['FP'])
        loss_epoch = sum(losses)/len(losses)
        

        if use_neptune:
            run["val/accuracy_epoch"].append(acc_epoch if math.isfinite(acc_epoch) else 0)
            run["val/losses_epoch"].append(loss_epoch if math.isfinite(loss_epoch) else 10)
            run["val/TPR_epoch"].append(TPR_epoch if math.isfinite(TPR_epoch) else 0)
            run["val/PPV_epoch"].append(PPV_epoch if math.isfinite(PPV_epoch) else 0)
        else:
            print('Acc  [epoch]: ', acc_epoch)
            print('Loss [epoch]: ', loss_epoch)

        # --- TPR at FPR = 0 -------------------------------------------------------

        # pick the smallest threshold that still gives zero false positives:
        neg_scores = y_scores[y_true == 0]
        pos_scores = y_scores[y_true == 1]

        if neg_scores.size > 0:
            # threshold just above the highest background score
            tau_fpr0 = np.nextafter(neg_scores.max(), np.inf)
            # by construction FPR is 0.0 with the >= rule
            tpr_at_fpr0 = float((pos_scores >= tau_fpr0).mean())
            fpr_at_tau  = 0.0
        else:
            # degenerate case: no background in validation
            tau_fpr0 = 1.0
            tpr_at_fpr0 = float((pos_scores >= tau_fpr0).mean())
            fpr_at_tau  = 0.0

        if use_neptune:
            run["val/TPR_at_FPR0"].append(tpr_at_fpr0)
            run["val/threshold_at_FPR0"].append(float(tau_fpr0))
            run["val/FPR_at_threshold"].append(float(fpr_at_tau))
        else:
            print(f"TPR@FPR=0: {tpr_at_fpr0:.4f} at τ={tau_fpr0:.6f}")
        
        # --- TPR at "≤ K false positives" -----------------------------------------

        K = 10  # max allowed false positives on validation

        if neg_scores.size > 0:
            s_desc = np.sort(neg_scores)[::-1]  # descending
            if K <= 0:
                tau_k = np.nextafter(s_desc[0], np.inf)  # > max neg -> 0 FPs
            elif K >= s_desc.size:
                tau_k = -np.inf  # accept all (≥ tau_k)
            else:
                # just above the (K-th) element in DESC order ensures FP_count ≤ K
                tau_k = np.nextafter(s_desc[K], np.inf)

            fp_mask = (neg_scores >= tau_k)
            tp_mask = (pos_scores >= tau_k)
            fp_count = int(fp_mask.sum())
            tpr_k    = float(tp_mask.mean()) if pos_scores.size else 0.0
            fpr_k    = float(fp_count / max(1, neg_scores.size))
            target_fpr_nominal = K / max(1, neg_scores.size)  # for reference
        else:
            tau_k, tpr_k, fpr_k, fp_count, target_fpr_nominal = 1.0, 0.0, 0.0, 0, 0.0

        if use_neptune:
            run["val/TPR_at_maxFP"].append(tpr_k)
            run["val/threshold_at_maxFP"].append(float(tau_k))
            run["val/FP_allowed_K"].append(int(K))
            run["val/FP_actual"].append(int(fp_count))
            run["val/FPR_at_threshold"].append(float(fpr_k))
            run["val/FPR_target_nominal"].append(float(target_fpr_nominal))
        else:
            print(f"TPR@FP≤{K}: {tpr_k:.4f}  |  τ={tau_k:.6g}  |  FP={fp_count}  |  FPR={fpr_k:.6g}")
# --------------------------------------------------------------------------



        ## best val epoch save
        if acc_epoch > best_val_acc:
            suffix = 'best_valacc_epoch.pt'
            best_val_acc = acc_epoch
            savename = None
            if use_neptune:
                savename = run["sys/id"].fetch() + suffix
            else:
                savename = 'ParT_modified' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + suffix
            # torch.save(model.state_dict(), '/users/alikaan.gueven/ParticleTransformer/PyTorchExercises/models/vtx_' + savename)
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + savename)
        
        if use_neptune:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + run["sys/id"].fetch() + '_epoch_' + str(epoch) + '.pt')
        else:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + '_epoch_' + str(epoch) + '.pt')


        ## min loss epoch save
        if loss_epoch < best_val_loss:
            suffix = 'best_valloss_epoch.pt'
            best_val_loss = loss_epoch
            savename = None
            if use_neptune:
                savename = run["sys/id"].fetch() + suffix
            else:
                savename = 'ParT_modified' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + suffix
            # torch.save(model.state_dict(), '/users/alikaan.gueven/ParticleTransformer/PyTorchExercises/models/vtx_' + savename)
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + savename)
        
        if use_neptune:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + run["sys/id"].fetch() + '_epoch_' + str(epoch) + '.pt')
        else:
            torch.save(model, '/groups/hephy/cms/alikaan.gueven/ParT/models/vtx_' + datetime.datetime.now().strftime('_%Y-%m-%d-%H-%M-%S_') + '_epoch_' + str(epoch) + '.pt')


        if use_neptune:
            isMatched = torch.cat(label_bucket)[:,1] == 1
            binVals1, binEdges1 = np.histogram(torch.cat(output_bucket)[:,1][isMatched].cpu(), np.arange(-0.01,1.01,0.02))
            binCenters1 = (binEdges1[:-1] + binEdges1[1:]) / 2

            binVals0, binEdges0 = np.histogram(torch.cat(output_bucket)[:,1][~isMatched].cpu(), np.arange(-0.01,1.01,0.02))
            binCenters0 = (binEdges0[:-1] + binEdges0[1:]) / 2
            
            plt.figure(figsize=(10,8))
            plt.errorbar(binCenters1, binVals1, np.sqrt(binVals1), fmt='o', capthick=1, capsize=3, color='indigo', markersize=3, label="matched")
            plt.errorbar(binCenters0, binVals0, np.sqrt(binVals0), fmt='o', capthick=1, capsize=3, color='red', markersize=3, label="unmatched")
            plt.ylim(1, 2*(sum(binVals1)+sum(binVals0)))
            plt.yscale('log')
            plt.title('Validation Dataset')
            plt.xlabel('ParT Score')
            plt.ylabel('vtx Count')
            plt.legend()
            fig_savepath = os.path.join(PROJECT_DIR, 'training/tmp/hist.png')
            plt.savefig(fig_savepath)

            run['score_hist'].append(neptune.types.File(fig_savepath))

    scheduler.step()
    gc.collect() # counter memory leaks at the end of each epoch

    # Check if the peak memory increases after each epoch!!
    # if torch.cuda.device_count():
    #     torch.cuda.empty_cache()      # ← releases the cached blocks
    
if use_neptune:
    run.stop()




