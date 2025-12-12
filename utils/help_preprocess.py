import numpy as np
import awkward as ak
import torch
from numba import njit
import math

def pad_and_fill(X, branchDict, svDim=12, tkDim=10, fillValue=float("nan")):
    def flatten_and_tensor(field, broadcast_to=None):
        ak_arr = X[field]
        if broadcast_to == 'sv':
            # Broadcast to any sv branch shape
            ak_arr, _ = ak.broadcast_arrays(ak_arr, X[branchDict['sv'][0]])
        else:
            pass
        flat = ak.flatten(ak_arr, axis=1)
        X_np = flat.to_numpy()
        return torch.tensor(X_np)

    X_dict = {}
    for field in X.fields:
        if branchDict['ev'] and field in branchDict['ev']:
            if field.startswith('n'):
                X[field] = ak.values_astype(X[field], np.int32)
            
            X_dict[field] = flatten_and_tensor(field, broadcast_to='sv')
        
        elif branchDict['sv'] and field in branchDict['sv']:
            X_dict[field] = flatten_and_tensor(field)
        elif branchDict['label'] and field in branchDict['label']:
            X_dict[field] = flatten_and_tensor(field)
        elif branchDict['lut'] and field == "SDVIdxLUT_TrackWeight":
            trIdx = X.SDVIdxLUT_TrackIdx
            svIdx = X.SDVIdxLUT_SecVtxIdx
            n_sv =  X.nSDVSecVtx
            
            builder = ak.ArrayBuilder()
            deepTable_weight(X[field], svIdx, n_sv, builder)
            deepX = builder.snapshot()

            almost_flat = ak.flatten(deepX, axis=1)
            padded = ak.pad_none(almost_flat, target=tkDim, clip=True, axis=1)
            filled = ak.fill_none(padded, 0., axis=1)

            X_np = filled.to_numpy()
            X_dict[field] = torch.tensor(X_np)
        elif branchDict['tk'] and field in branchDict['tk']:
            trIdx = X.SDVIdxLUT_TrackIdx
            svIdx = X.SDVIdxLUT_SecVtxIdx
            n_sv =  X.nSDVSecVtx
            
            builder = ak.ArrayBuilder()
            deepTable(X[field], trIdx, svIdx, n_sv, builder)
            deepX = builder.snapshot()
            # print(deepX.type)
            # print('deepX: ', deepX)

            field_fillValue = {
                'SDVTrack_E':  0.,
                'SDVTrack_pz': 0.
            }.get(field, fillValue)         # Returns 'fillValue' if the 'field' does not exist.

            almost_flat = ak.flatten(deepX, axis=1)
            # print('almost_flat: ', almost_flat)
            padded = ak.pad_none(almost_flat, target=tkDim, clip=True, axis=1)
            filled = ak.fill_none(padded, field_fillValue, axis=1)

            X_np = filled.to_numpy()
            X_dict[field] = torch.tensor(X_np)

    return X_dict


@njit
def deepTable(tkBranch, trIdx, svIdx, n_sv, builder):
    """
    Takes the track level branch and converts its shape
    from: (nEvent * var * float32) to (nEvent * var * var * float32)
    representing the association of each tk with sv.

    svIdx: [[0, 0, 1, 1,  2, 2,  3,  3, ...], ...]
    trIdx: [[1, 3, 3, 28, 7, 8, 10, 11, ...], ...]
    n_sv:  [4, 0, 6, 1, 1, 7, ...]
    """
    # at event level depth already
    # every element added are at event level depth
    for ev in range(len(n_sv)):                   # 
        builder.begin_list()                      # adding sv level depth
        for sv in range(n_sv[ev]):                # for each vtx ... 
            builder.begin_list()
            for i2, col in enumerate(svIdx[ev]):  # getting an svIdx from the svIdxs for that event
                if col == sv:                     # if the same svIdx 
                    builder.append(tkBranch[ev][trIdx[ev][i2]])
            builder.end_list()
        builder.end_list()

@njit
def deepTable_weight(wgtBranch, svIdx, n_sv, builder):
    for ev in range(len(n_sv)):
        builder.begin_list()
        for sv in range(n_sv[ev]):
            builder.begin_list()
            for i2, col in enumerate(svIdx[ev]):     # same walk through LUT
                if col == sv:
                    builder.append(wgtBranch[ev][i2])  # ← no trIdx here
            builder.end_list()
        builder.end_list()



# def delta_phi_abs(phi1, phi2):
#     """
#     |Δφ| computed as arccos(cos(φ₁ - φ₂)).
#     Works on Awkward arrays / NumPy scalars alike.
#     Result is in [0, π].
#     """
#     return np.arccos(np.cos(phi1 - phi2))


def calc_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta):
    builder = ak.ArrayBuilder()
    n_ev =  ak.num(v_phi, axis=0)
    n_vtx = ak.num(v_phi, axis=1)
    n_jet = ak.num(j_phi, axis=1)
    vertex_min_dR_deta_dphi(v_phi, v_eta,
                            j_phi, j_eta,
                            int(n_ev), n_vtx, n_jet, builder)
    arrs = builder.snapshot()
    best_dR   = arrs[...,0]
    best_dphi = arrs[...,1]
    best_deta = arrs[...,2]
    return (best_dR, best_dphi, best_deta)

@njit
def vertex_min_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta, n_ev, n_vtx, n_jet, builder):
    for ev in range(n_ev):
        nv = n_vtx[ev]
        nj = n_jet[ev]
        builder.begin_list()
        for vtx in range(nv):
            builder.begin_list()
            best_dR   = None
            best_dphi = None
            best_deta = None
            for jet in range(nj):
                dphi = math.acos(math.cos(j_phi[ev][jet] - v_phi[ev][vtx]))
                deta = abs(j_eta[ev][jet] - v_eta[ev][vtx])
                dR   = math.hypot(dphi, deta)
                if (best_dR is None) or (dR < best_dR):
                    best_dR = dR
                    best_dphi = dphi
                    best_deta = deta

            builder.append(best_dR)
            builder.append(best_dphi)
            builder.append(best_deta)
            builder.end_list()                     # ── end vertex list
        builder.end_list()                         # ── end event list


# def vertex_min_dR_deta_dphi(v_phi, v_eta, j_phi, j_eta):
#     """
#     """
# 
#     # broadcast vertices against jets  →  (event, vertex, jet)
#     j_phi3 = j_phi[:, None, :]             # (E, 1, J)
#     j_eta3 = j_eta[:, None, :]
#     v_phi3 = v_phi[:, :, None]             # (E, V, 1)
#     v_eta3 = v_eta[:, :, None]
#     
# 
#     # Δφ, Δη, ΔR
#     dphi = delta_phi_abs(j_phi3, v_phi3)   # |Δφ|  (E,V,J)
#     deta = np.abs(j_eta3 - v_eta3)         # |Δη|
#     dR   = np.sqrt(dphi**2 + deta**2)      # ΔR
# 
#     # nearest-jet index for every vertex
#     idx = ak.argmin(dR, axis=-1)
#     print('dR: ', dR.type)
#     print('idx: ', idx.type)
#     min_dR = dR[idx[:,None,:]]
# 
#     min_dR, min_dphi, min_deta = map(gather, (dR, dphi, deta))
#     print('min_dR: ', min_dR.type)
#     exit()
#     return (min_dR[...,0], min_dphi[...,0], min_deta[...,0])



def probe_shapes(my_dataset_iterator, data_dict, branch_dict, preprocess_fn, step_size=32):
    """
    Returns tensor shapes after preprocessing.

    Parameters
    ----------
    my_dataset_iterator : iterator
        Dataset iterator, e.g. ModifiedUprootIterator
    data_dict : dict
        Something like this: trainDict = {'sig': trainSigFileList, 'bkg': None}
    branch_dict : dict
        Dict of branches where the keys are 'ev', 'sv', 'tk', 'label', 'lut'.
    preprocess_fn : callable
        Function to process awkward arrays to tensors.
    step_size : int, optional

    Returns
    -------
    dict
        A dict of torch tensor shapes
    """
    ds = my_dataset_iterator(data_dict, branch_dict, shuffle=False, nWorkers=0, step_size=step_size)
    dl = torch.utils.data.DataLoader(ds, num_workers=0, collate_fn=preprocess_fn, drop_last=True)
    batch = next(iter(dl))
    return {k: tuple(v.shape) for k, v in batch.items()}



def add_ngoodtrack_with_deepTable(X):
    """
    Reimplement:
      SDVTrack_isGoodTrack = ...
      SDVSecVtx_ngoodtk    = SDVSecVtx_nGoodTrack(...)

    using Awkward + deepTable.

    Adds to X:
      - SDVTrack_isGoodTrack      (per-track bool)
      - SDVSecVtx_ngoodTrack      (per-SV int)
    """

    SDVTrack_ptSig      = X['SDVTrack_pt'] / X['SDVTrack_ptError']
    SDVTrack_dxySig     = np.abs(X['SDVTrack_dxy'])/ X['SDVTrack_dxyError']
    SDVTrack_dxydzratio = np.abs(X['SDVTrack_dxy'] / X['SDVTrack_dz'])


    SDVTrack_isGoodTrack = (
        (X["SDVTrack_normalizedChi2"]    < 5)   &
        (X["SDVTrack_numberOfValidHits"] > 13)  &
        ((1.0 / SDVTrack_ptSig)     < 0.015)    &
        (SDVTrack_dxySig            > 4)        &
        (SDVTrack_dxydzratio        > 0.25)     &
        (X["SDVTrack_pfRelIso03_all"]  < 5)
    )

    trIdx = X["SDVIdxLUT_TrackIdx"]   # (event, nAssoc)
    svIdx = X["SDVIdxLUT_SecVtxIdx"]  # (event, nAssoc)
    n_sv  = X["nSDVSecVtx"]           # (event,)

    builder = ak.ArrayBuilder()
    deepTable(SDVTrack_isGoodTrack, trIdx, svIdx, n_sv, builder)
    isGood_deep = builder.snapshot()  # shape: (nEvent, nSV, nTracksPerSV), dtype=bool

    
    X["SDVSecVtx_ngoodTrack"] = ak.sum(isGood_deep, axis=-1)  # (nEvent, nSV)
    return X



def build_event_filter(X):
    """
    Build the event-level filter mask, like:
      - ngoodTrack >= 1
      - preselection (MET, flags, jets, HEM veto, leptons, etc.)

    Parameters
    ----------
    X : ak.Array

    Returns
    -------
    event_mask : ak.Array (bool, shape (nEvents,))
        Boolean mask to select events passing the full preselection.
    """


    # Vertex filter
    # ------------------------------------------------------------------
    X = add_ngoodtrack_with_deepTable(X)
    # per_sv_mask = X["SDVSecVtx_ngoodTrack"] >= 1
    per_sv_mask = X["SDVSecVtx_ngoodTrack"] >= 1
    event_mask_svs = ak.any(per_sv_mask, axis=1)   # (nEvents,)


    # Preselection filter
    # ------------------------------------------------------------------
    jet_selHEM = (X["Jet_jetId"] == 6)

    nJetHEM = ak.sum(
        jet_selHEM
        & (X["Jet_eta"] > -3.0)
        & (X["Jet_eta"] < -1.3)
        & (X["Jet_phi"] > -1.57)
        & (X["Jet_phi"] < -0.87),
        axis=1,
    )

    jet_sel = (
        (X["Jet_jetId"] == 6)           &
        (np.abs(X["Jet_eta"]) < 2.4)    &
        (X["Jet_neHEF"] < 0.8)          &
        (X["Jet_chHEF"] > 0.1)          &
        (X["Jet_pt"] > 20)
    )

    Jet_pt_sel  = X["Jet_pt"][jet_sel]
    Jet_phi_sel = X["Jet_phi"][jet_sel]
    nJet_sel = ak.sum(jet_sel, axis=1)


    leadingjet_pt  = ak.fill_none(ak.firsts(Jet_pt_sel),  0.0)
    leadingjet_phi = ak.fill_none(ak.firsts(Jet_phi_sel), 0.0)

    dphi_MET_jet0 = np.arccos(np.cos(leadingjet_phi - X["MET_phi"]))

    nMuon_sel = ak.sum(
        (X["Muon_looseId"] > 0) &
        (X["Muon_eta"] < 2.4)   &
        (X["Muon_pt"] > 10),
        axis=1,
    )


    nEle_sel = ak.sum(
        (X["Electron_cutBased"] >= 1)   &
        (X["Electron_eta"] < 2.5)       &
        (X["Electron_pt"] > 10),
        axis=1,
    )

    nTau_sel = ak.sum(
        (X["Tau_idDecayModeOldDMs"] >= 1)   &
        (X["Tau_eta"] < 2.3)                &
        (X["Tau_pt"] > 18),
        axis=1,
    )

    nPhoton_sel = ak.sum(
        (X["Photon_cutBased"] >= 1)     &
        (X["Photon_eta"] < 2.5)         &
        (X["Photon_pt"] > 15),
        axis=1,
    )


    presel = (
        # (X["MET_pt"] > 200)
        X["Flag_goodVertices"]
        & X["Flag_globalSuperTightHalo2016Filter"]
        & X["Flag_HBHENoiseFilter"]
        & X["Flag_HBHENoiseIsoFilter"]
        & X["Flag_EcalDeadCellTriggerPrimitiveFilter"]
        & X["Flag_BadPFMuonFilter"]
        & X["Flag_BadPFMuonDzFilter"]
        & X["Flag_hfNoisyHitsFilter"]
        & X["Flag_BadChargedCandidateFilter"]
        & X["Flag_eeBadScFilter"]
        & X["Flag_ecalBadCalibFilter"]
        & X["HLT_PFMETNoMu120_PFMHTNoMu120_IDTight"]
        & (nMuon_sel == 0)
        & (nEle_sel == 0)
        & (nTau_sel == 0)
        & (nPhoton_sel == 0)
        & (nJet_sel > 0)
        # & (leadingjet_pt > 100)
        & (nJetHEM == 0)
        & (dphi_MET_jet0 > 1.0)
    )

    # combine vertex sel + presel
    event_mask = event_mask_svs & presel
    return event_mask

