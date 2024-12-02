# Configs for training the model

# Inherit configs from other config file

import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from Data_preprocsessing_config import SEG_NUM_TIMESTEPS, FACTORS_NOT_USED_FOR_FM, VERSION

# GPU
GPU_NAME = "cuda:0"

# General model structure
INPUT_LENGTH = SEG_NUM_TIMESTEPS
INPUT_LENGTH_FFT = (INPUT_LENGTH + 2) // 2

# Model structure for WSC pipeline method
epochs = 60
rTrain = 0.8;
rTest = 0.1;
batch_size = 32
num_bins = 40
coef_delta = 0

ae_struct_list = {2:[202, 40, 20], 3:[202, 20, 20], 4:[202, 40]}
train_ae_idx = [False, False, True, True, True]

dt2ind = {'glitch_H':0, 'glitch_L':1, 'noise':2, 'BBH':3, 'SGHF':4}
ind2dt = {}
