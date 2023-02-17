#import opensmile
#import hearbaseline
#import openl3
#import tensorflow_hub as hub
import os
from byol_a.common import load_yaml_config
from resemblyzer import VoiceEncoder
from speechbrain.pretrained import EncoderClassifier
from byol_a.augmentations import PrecomputedNorm
from byol_a.models.audio_ntt import AudioNTT2020
from byol_a.models.cvt import CvT
from byol_a.models.clstm import CLSTM
from byol_a.models.resnetish import resnetish34
import torch
import multiprocessing
"""Main constants used in the code."""

_RANDOM_SEED = 42
_REQUIRED_SAMPLE_RATE = 16000  # Hz
_SCORING = 'recall_macro'
NUM_WORKERS = 0 # int(multiprocessing.cpu_count() / (max(1, torch.cuda.device_count())))
MAX_WINS = 40 # use 20 for 3s chunk, use 40 for 5s chunks use at leat 550 if not clip 
CLIP_LEN = 5 # clip to length (second)
_EMBED_PATH = '/home/zwu1/hdd/data/VSA_Embed_byols'

_VSA_PATH = '/home/zwu1/data/VSA/'
_COG1_LOAD_PATH = os.path.join(_VSA_PATH, 'CogLoadv1')
_COG2_LOAD_PATH = os.path.join(_VSA_PATH, 'CogLoadv2')
_COG3_LOAD_PATH = os.path.join(_VSA_PATH, 'audio_visual_stress_data')
_COG4_LOAD_PATH = os.path.join(_VSA_PATH, 'bic_dataset')
_PHY_LOAD_PATH = os.path.join(_VSA_PATH, 'PhyLoad/PhyLoad_utterances')


# Pre-normalization Mean and SD statistics for each classification dataset
# Format: 'dataset_name': [log_spectrogram_mean_value, log_spectrogram_sd_value]
_CLF_STATS_DICT = {
    'cog1': [-4.198391437530518, 3.8650004863739014],
    'cog1_nn': [-4.035102367401123, 3.838627576828003],
    'cog2': [-7.603394508361816, 4.616683483123779],
    'cog2_nn': [-7.508063316345215, 4.649827480316162],
    'cog3': [-6.787594795227051, 3.6053149700164795],
    'cog3_nn': [-6.809290885925293, 3.6053626537323],
    'cog4': [-7.588655471801758, 3.2360992431640625],
    'phy': [-3.10797119140625, 2.584670066833496],
    'phy_nn': [-3.050205707550049, 2.6070942878723145],
    'allcog': [-6.543684482574463, 3.9297783374786377],
    'all': [-6.1996002197265625, 3.7950656414031982], # stats for 75% split training data
    'personalized': [-6.23012638092041, 3.801431655883789],
    'all_split': [-6.210606098175049, 3.808908224105835] # stats for 60% split training data
}


# Define All models DSP-based and Data-driven based
_MODELS_DICT = {
                'BYOL-S_cvt_hybrid': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                    s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),
                # 'BYOL-S_cvt_hybrid_randaug_randclip': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),
                # 'BYOL-S_cvt_hybrid_randaug4_randclip': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),
                # 'BYOL-S_cvt_hybrid_randaug4_randclip_88': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),
                # 'BYOL-S_cvt_hybrid_gasser': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),
                # 'BYOL-S_cvt_hybrid_gasser_halfbyols': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),  
                # 'BYOLS_cvt_randaug_unsync': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='mean+max'),  
                # 'Timestamp_cvt_randaug_unsync': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid_randaug_randclip': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid_randaug4_randclip': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid_randaug4_randclip_88': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid_gasser': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),
                # 'Timestamp_cvt_hybrid_gasser_halfbyols': CvT(s1_emb_dim=64,s1_depth=1,s1_mlp_mult=4,s2_emb_dim=256,s2_depth=1,
                #     s2_mlp_mult=4,s3_emb_dim=512,s3_depth=1,s3_mlp_mult=4,pool='timestamp'),   
                }


_WEIGHT_DICT = {
    'BYOL-S_cvt_hybrid': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42.pth',
    'BYOL-S_cvt_hybrid_randaug_randclip': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42-aug-mix1d_k=2_mix2d_k=0.5.pth',
    'BYOL-S_cvt_hybrid_randaug4_randclip': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42-aug-mix1d_k=4_mix2d_k=0.5.pth',
    'BYOL-S_cvt_hybrid_randaug4_randclip_88': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS88-e100-bs256-lr0003-rs42-aug-mix1d_k=4_mix2d_k=0.5.pth',
    'BYOL-S_cvt_hybrid_gasser': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth',
    'BYOL-S_cvt_hybrid_gasser_halfbyols': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandhalfbyolaloss6373-e100-bs256-lr0003-rs42.pth',
    'BYOLS_cvt_randaug_unsync': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL0-e100-bs128-lr0003-rs42-aug-mix1d_k=2_mix2d_k=0.5_unsync.pth',
    'Timestamp_cvt_hybrid': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42.pth',
    'Timestamp_cvt_hybrid_randaug_randclip': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42-aug-mix1d_k=2_mix2d_k=0.5.pth',
    'Timestamp_cvt_hybrid_randaug4_randclip': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS6373-e100-bs256-lr0003-rs42-aug-mix1d_k=4_mix2d_k=0.5.pth',
    'Timestamp_cvt_hybrid_randaug4_randclip_88': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL1-OS88-e100-bs256-lr0003-rs42-aug-mix1d_k=4_mix2d_k=0.5.pth',
    'Timestamp_cvt_hybrid_gasser': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth',
    'Timestamp_cvt_hybrid_gasser_halfbyols': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandhalfbyolaloss6373-e100-bs256-lr0003-rs42.pth',
    'Timestamp_cvt_randaug_unsync': '/home/zwu1/hdd/checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-SL0-e100-bs128-lr0003-rs42-aug-mix1d_k=2_mix2d_k=0.5_unsync.pth'
}

_SPEAKER_EMBED = {
    'resemblyzer': VoiceEncoder(),
    'ecapa': EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})
}

_MLP_GRID  = {
    "clf": ['mlp'],
    "hidden_layers": [1],
    "hidden_dim": [512],
    "weight_decay": [1e-4],
    "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5],
    #"lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5],
    # "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5, 1e-5],
    # "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "patience": [30],
    "max_epochs": [300],
    "min_epochs": [100],
    "check_val_every_n_epoch": [1],
    "initialization": [torch.nn.init.xavier_normal_],
    "gpus": [1]
}

_TRANSF_GRID  = {
    "clf": ['transf'],
    "hidden_layers": [2],
    "hidden_dim": [64],
    "weight_decay": [1e-4],
    "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5],
    #"lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5],
    # "lr": [3.2e-3, 1e-3, 3.2e-4, 1e-4, 3.2e-5, 1e-5],
    # "lr": [1e-2, 3.2e-3, 1e-3, 3.2e-4, 1e-4],
    # "lr": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    "patience": [30],
    "max_epochs": [300],
    "min_epochs": [100],
    "check_val_every_n_epoch": [1],
    "initialization": [torch.nn.init.xavier_normal_],
    "gpus": [1]
}

_CLF_GRID = {
    "mlp": _MLP_GRID,
    "transf": _TRANSF_GRID
}

