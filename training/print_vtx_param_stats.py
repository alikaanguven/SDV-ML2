import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import networks.ParT_K as ParT
import utils.network_helpers as nh


model = ParT.ParticleTransformerDVTagger(input_dim=7,
                                         input_svdim=8,
                                         num_classes=2,
                                         pair_input_dim=4,
                                         embed_dims=[128, 512, 128],
                                         for_inference=False,
                                         block_params={'dropout': 0.20,
                                                       'attn_dropout': 0.15,
                                                       'activation_dropout': 0.15},
                                         use_amp=False,
                                         num_layers=4)

stats = nh.parameter_stats(model)
print(f"Total params        : {stats['total']:,}")
print(f"  Trainable         : {stats['trainable']:,}")
print(f"    • weights       : {stats['trainable_weights']:,}")
print(f"    • biases        : {stats['trainable_biases']:,}")
print(f"    • other         : {stats['trainable_other']:,}")
print(f"  Non-trainable     : {stats['non_trainable']:,}\n")
