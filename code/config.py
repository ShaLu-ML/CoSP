
import os
from runtime_paths import explicit_paths
from experiment_settings import load_experiment_settings, FINAL_INFO_COLUMNS
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import torch

# <editor-fold desc="------ settings ------">
COMPUTER = 'custom'
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VERBOSE = 3
# </editor-fold>

# <editor-fold desc="------ CONFIG ------">
# Participant metadata and saved-model mappings belong in a private JSON file.
_private = load_experiment_settings(os.environ.get('COSP_EXPERIMENT_CONFIG'))
PAT_LIST_ALL = [row['pat'] for row in _private['participants']]
PAT_ALL_DAYS = [row['days'] for row in _private['participants']]
PID_XRAY = _private['pid_xray']
XRAY_FILES = _private['xray_files']
columns = list(FINAL_INFO_COLUMNS)
# Keep the expected columns even when no private study is configured.
FINAL_INFO = pd.DataFrame(_private['final_info']).reindex(
    columns=list(dict.fromkeys(columns + [key for row in _private['final_info'] for key in row])))

# get all the model_ids used
MIDS_USED = []
m_names = [name for name in FINAL_INFO.columns if name.startswith('model')]
for m_type in m_names:
    mid = FINAL_INFO[m_type].dropna().tolist()
    MIDS_USED.extend(mid)

# get all the preictal interval tested in minutes
PRE_ITVL_TESTED = [int(name[5:]) for name in FINAL_INFO.columns if name.startswith('model')]

# patients selected for the experiments
PAT_LIST = FINAL_INFO['pat'].to_list()
PAT_ID_LIST = FINAL_INFO['pat_id'].to_list()
PAT_DICT = dict(zip(PAT_ID_LIST, PAT_LIST))

# all support threshold approaches: tst is oracle in paper.
THD_APPROACHES = ['val', 'tst', 'dynamic', 'fixed']

EEG_FREQ = 400
EEG_CHN = 16
REC_LEN_MIN = 1
N_CHN_PAIR = 120
N_FREQ = 109   # number of decrete frequency for coherence computation

# Define frequency bands
FREQ_BANDS = {
    'delta': (0, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma1': (30, 100),
    'gamma2': (100, 170),
    # 'gamma': (30, 170)
}


# ------------------------------------------ ALL --------------------------------------------------
idx_run = 100 # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
rank = 0 # 0-3
if torch.cuda.is_available(): torch.cuda.set_device(rank)
# DEVICE = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
DEVICE = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

# BATCH_RUN = False   #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TEST_ONLY = True
# BASE_MODEL =  {'model_id':1, 'round_id':1} # -1
# if TEST_ONLY == False: BASE_MODEL = -1
# if BATCH_RUN == True: BASE_MODEL = -1 # invalid when BATCH_RUN is True

LOSS_PP_WEIGHT = 0.5
PP_SCALE_FACTOR = 1
# --------------------------------------------------------------------------------------------
f_log = f'output{idx_run}.log'
TEST = False



CONFIG ={
    #------  training related ------
    # model related
    'model': 'CoSP',
    # 'hidden_size':  [512, 256, 128, 64], # [512, 256, 128, 64], # CoSP: [32, 64, 128, 256]
    'hidden_size': [512, 256, 128, 64], # [512, 256, 128, 64], [256, 128]
    'kernel_size':  [7, 5, 3, 3], # CoSP: [7, 5, 3, 3]

    # training data related
    'trn_label': 'both',  # inter, pre, both, ictal
    'trn_sz_pct': 0.6,
    'val_sz_pct': 0.2,
    'dropout_thd': 10,

    # parameters for ViT model,  model_type=ViT
    'vit_patch_size': 8,  # patch size
    'vit_image_size': 128,
    'vit_in_channels': 1,

    # training process related
    'batch_size': 32,
    'learning_rate': 1e-6,
    'lr_decay_step': 0,
    'lr_decay_gamma': 0.5,
    'n_epochs': 200,
    'patience': 20,  # early stop patience
    'gpu_mode': 'gpu',  # ddp-3, gpu, ddp

    #------  evaluate related ------
    'test_inter_itvl':  1 * 24 * 60, # in minutes, options: -1,  1 * 24 * 60
    # 'test_with_config': False,
    # 'test_model': BASE_MODEL,

    #------  data processing related ------
    'pat_id': None,  # select a participant from the private configuration
    'pred_itvl_min': 1,  # prediction interval in minutes, default 1 minute
    'pre_itvl_min': 45,  # CoSP: 4, 15, 30, 45 60, ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    'inter_itvl_min': 2 * 24 * 60, #-1,  2 * 24 * 60
    'seg_len_sec': 10,
    'n_seg_limit': 15000,  # CoSP: 15000, the maximum number of interictal segments
    'pre_sample': 'fixed',  # CoSP:'fixed'
    'inter_sample': '1segNmin-10',  # CoSP: '1segNmin-10' or 'non-overlap'
}



CONFIG.update(_private['config'])
if CONFIG['pat_id'] is not None and CONFIG['pat_id'] not in PAT_ID_LIST:
    raise ValueError("CONFIG['pat_id'] must appear in the private final_info mapping")

_explicit_paths = explicit_paths(os.environ)
if _explicit_paths is None:
    raise ValueError('Set COSP_DATA_PATH, COSP_WORK_PATH, COSP_FILE_LEN_MIN and '
                     'COSP_REC_OFFSET; see docs/setup.md')
DATA_PATH, WORK_PATH, CONFIG['file_len_min'], REC_OFFSET = _explicit_paths
BASE_PATH = DATA_PATH.parent
WORK_PATH.mkdir(parents=True, exist_ok=True)

assert (CONFIG['trn_sz_pct'] + CONFIG['val_sz_pct']) < 1
assert CONFIG['trn_label'] in ['pre', 'inter', 'both', 'ictal']
# </editor-fold>


# <editor-fold desc="------ path ------">


INFO_PATH = WORK_PATH /'datainfo'
INFO_PATH.mkdir(parents=False, exist_ok=True)

ANNO_PATH = INFO_PATH / 'annotation'
ANNO_PATH.mkdir(parents=False, exist_ok=True)

# MODEL_PATH = WORK_PATH / "model"
MODEL_PATH = WORK_PATH / f"model"
MODEL_PATH.mkdir(parents=False, exist_ok=True)

RST_PATH = WORK_PATH / 'result'
RST_PATH.mkdir(parents=True, exist_ok=True)
(RST_PATH/f"detail").mkdir(parents=True, exist_ok=True)

STAT_PATH = WORK_PATH / f"datainfo/statistics"
STAT_PATH.mkdir(parents=True, exist_ok=True)

LIME_PATH = WORK_PATH / f"explain_lime"
LIME_PATH.mkdir(parents=True, exist_ok=True)
