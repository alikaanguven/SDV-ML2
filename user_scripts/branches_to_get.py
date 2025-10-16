def get_branchDict():
    branchDict = {}
    branchDict['ev'] = ['MET_phi',
                        'MET_pt',
                        'nSDVSecVtx',
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
                        'SDVTrack_eta',
                        'SDVTrack_phi',
                        'SDVTrack_dxy',
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

    # branchDict['jet'] = ['Jet_phi', 
    #                     'Jet_eta',
    #                     'Jet_jetId',
    #                     'Jet_pt',
    #                     'Jet_chEmEF',
    #                     'Jet_neEmEF',
    #                     'Jet_muonIdx1',
    #                     'Jet_muonIdx2'
    #                     ]

    branchDict['label'] = ['SDVSecVtx_matchedLLPnDau_bydau']


    return branchDict