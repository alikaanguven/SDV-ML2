import sys
sys.path.append("/users/alikaan.gueven/SDV-ML/ParticleTransformer/SDV-ML")

import torch
import awkward as ak
import numpy as np
import utils.help_preprocess as hp


def transform(batch, branch_dict):
    """
    Preprocess the input data for training or evaluation.
    
    Args:
        
        
    Returns:
        
    """
    out_dict = {}
    X = ak.concatenate(batch, axis=0)
    # X = batch[0]

    # X['jet0_phi'] = X['Jet_phi'][:, 0]
    # X['jet0_eta'] = X['Jet_eta'][:, 0]
    # 
    # Jet_isTight = ak.values_astype(X['Jet_jetId'] & (1 << 1), bool)
    # jet_sel_vetomapsel = (Jet_isTight &
    #                      (X['Jet_pt'] > 15) &
    #                      ((X['Jet_chEmEF'] + X['Jet_neEmEF']) < 0.9) &
    #                      (X['Jet_muonIdx1'] == -1) &
    #                      (X['Jet_muonIdx2'] == -1)
    #                      )
    # Jet_phi_vetomapsel = X.Jet_phi[jet_sel_vetomapsel]
    # Jet_eta_vetomapsel = X.Jet_eta[jet_sel_vetomapsel]
    # X['Jet_phi_vetomapsel'] = Jet_phi_vetomapsel
    # X['Jet_eta_vetomapsel'] = Jet_eta_vetomapsel
    # 
    # 
    # min_dR, min_dphi, min_deta = hp.calc_dR_deta_dphi(X['SDVSecVtx_L_phi'], X['SDVSecVtx_L_eta'], X['Jet_phi_vetomapsel'], X['Jet_eta_vetomapsel'])
    # X['SDVSecVtx_closestJetdR']   = min_dR
    # 
    # branch_dict['ev'].append('jet0_phi')
    # branch_dict['ev'].append('jet0_eta')
    # branch_dict['sv'].append('SDVSecVtx_closestJetdR')
    # branch_dict['sv'].append('SDVSecVtx_closestJetdphi')
    # branch_dict['sv'].append('SDVSecVtx_closestJetdeta')
    
    X = hp.pad_and_fill(X, branch_dict, svDim=12, tkDim=10, fillValue=float("nan"))

    X['SDVTrack_vtxdR'] = torch.abs(X['SDVTrack_eta'] - X['SDVSecVtx_L_eta'][...,np.newaxis])**2 + \
                          torch.arccos(torch.cos(X['SDVTrack_phi'] - X['SDVSecVtx_L_phi'][...,np.newaxis]))**2

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------
    for key in X.keys(): X[key] = X[key].unsqueeze(0)




    out_dict["tk_pair_features"] = torch.cat((X['SDVTrack_px'],
                                              X['SDVTrack_py'],
                                              X['SDVTrack_pz'],
                                              X['SDVTrack_E']), dim=0).permute(1,0,2)

    eps = 1e-4

    out_dict["tk_features"] = torch.cat((
                                         torch.log(X['SDVTrack_pt']),
                                         X['SDVTrack_eta'],
                                         X['SDVTrack_vtxdR'],
                                         X['SDVTrack_dxy'] / (X['SDVTrack_dz'] + eps),
                                         # X['SDVTrack_dz'],
                                         X['SDVTrack_normalizedChi2'],
                                         torch.log(eps+ X['SDVTrack_pfRelIso03_all']),
                                         X['SDVIdxLUT_TrackWeight'],
                                         torch.cos(X['SDVSecVtx_L_phi'][...,np.newaxis] - X['SDVTrack_phi']),
                                         # torch.cos(X['MET_phi'][...,np.newaxis] - X['SDVTrack_phi'])
                                         ), dim=0).permute(1,0,2)
    

    out_dict["tk_mask"] = (~torch.isnan(out_dict["tk_features"][:,0:1,:])).to(torch.bool)   # 1 = normal, 0 = NaN


    
    out_dict["sv_features"] = torch.cat((
                                         torch.log(X['MET_pt']),
                                         torch.log(X['SDVSecVtx_pt']),
                                         X['SDVSecVtx_L_eta'],
                                         # X['SDVSecVtx_L_phi'],
                                         torch.log(X['SDVSecVtx_LxySig']),
                                         X['SDVSecVtx_pAngle'],
                                         X['SDVSecVtx_charge'],
                                         X['SDVSecVtx_chi2'] / (X['SDVSecVtx_ndof']),
                                         X['SDVSecVtx_sum_tkW'] / (X['SDVSecVtx_tracksSize']),
                                         # X['SDVSecVtx_closestJetdR'],
                                         # torch.cos(X["MET_phi"] - X["SDVSecVtx_L_phi"]),
                                         # torch.cos(X["jet0_phi"] - X["SDVSecVtx_L_phi"]),
                                         # torch.abs(X["jet0_eta"] - X["SDVSecVtx_L_eta"]),
                                         ), dim=0).permute(1,0)[..., np.newaxis]

    out_dict["label"] = X['SDVSecVtx_matchedLLPnDau_bydau'].permute(1,0)
    
    torch.nan_to_num(out_dict["tk_pair_features"], nan=-9.9, out=out_dict["tk_pair_features"])
    torch.nan_to_num(out_dict["tk_features"],      nan=-9.9, out=out_dict["tk_features"])
    torch.nan_to_num(out_dict["sv_features"],      nan=-9.9, out=out_dict["sv_features"])


    return out_dict