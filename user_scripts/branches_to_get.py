def get_branchDict(isData=False):
    branchDict = {}
    branchDict['ev'] = ['MET_phi',
                        'MET_pt',
                        'nSDVSecVtx'
                        ]
    
    branchDict['sv'] = ['SDVSecVtx_pt', 
                        'SDVSecVtx_pAngle', 
                        'SDVSecVtx_charge', 
                        'SDVSecVtx_ndof', 
                        'SDVSecVtx_chi2', 
                        'SDVSecVtx_tracksSize', 
                        'SDVSecVtx_sum_tkW', 
                        'SDVSecVtx_LxySig', 
                        'SDVSecVtx_L_phi', 
                        'SDVSecVtx_L_eta', 
                        ]

    branchDict['tk'] = ['SDVTrack_pt',
                        'SDVTrack_ptError',
                        'SDVTrack_eta',
                        'SDVTrack_phi',
                        'SDVTrack_dxy',
                        'SDVTrack_dxyError',
                        'SDVTrack_dz', 
                        'SDVTrack_normalizedChi2',
                        # 'SDVTrack_dr03TkSumPt',
                        'SDVTrack_pfRelIso03_all',
                        'SDVTrack_numberOfValidHits',
                        'SDVTrack_validFraction',
                        ]

    branchDict['lut'] = ['SDVIdxLUT_SecVtxIdx', 
                        'SDVIdxLUT_TrackIdx',
                        'SDVIdxLUT_TrackWeight'
                        ]

    branchDict['jet'] = ['Jet_phi', 
                        'Jet_eta',
                        'Jet_jetId',
                        'Jet_pt',
                        'Jet_chEmEF',
                        'Jet_neEmEF',
                        'Jet_muonIdx1',
                        'Jet_muonIdx2',
                        'Jet_neHEF',
                        'Jet_chHEF',
                        ]
    
    branchDict['filter'] = [# 'SDVSecVtx_ngoodTrack',
                            'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
                            'Flag_goodVertices',
                            'Flag_globalSuperTightHalo2016Filter',
                            'Flag_HBHENoiseFilter',
                            'Flag_HBHENoiseIsoFilter',
                            'Flag_EcalDeadCellTriggerPrimitiveFilter',
                            'Flag_BadPFMuonFilter',
                            'Flag_BadPFMuonDzFilter',
                            'Flag_hfNoisyHitsFilter',
                            'Flag_BadChargedCandidateFilter',
                            'Flag_eeBadScFilter',
                            'Flag_ecalBadCalibFilter',
                            ]
    
    branchDict['mu'] = ['Muon_looseId',
                        'Muon_eta',
                        'Muon_pt',
                        ]

    branchDict['ele'] = ['Electron_cutBased',
                         'Electron_eta',
                         'Electron_pt',
                        ]


    branchDict['tau'] = ['Tau_idDecayModeOldDMs',
                         'Tau_eta',
                         'Tau_pt',
                         ]

    
    branchDict['pho'] = ['Photon_cutBased',
                         'Photon_eta',
                         'Photon_pt',
                         ]
    
    branchDict['ngtk'] = ['SDVTrack_ptError',
                          ]

    if isData:
        branchDict['label'] = []
    else:
        branchDict['label'] = ['SDVSecVtx_matchedLLPnDau_bydau']


    return branchDict