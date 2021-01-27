from pathlib import Path
import re
import torch
import torchaudio
import numpy as np
from python_speech_features import mfcc
import sys
sys.path.append("..")
from config.path_list import *
from config.speaker_list import *
from hyper_params import hyper_params

FEATURE_PATH.mkdir(parents=True, exist_ok=True)
(FEATURE_PATH / "train").mkdir(parents=True, exist_ok=True)
(FEATURE_PATH / "val").mkdir(parents=True, exist_ok=True)
(FEATURE_PATH / "eval").mkdir(parents=True, exist_ok=True)

    
"""
f_info[0] : speaker ID
f_info[1] : session number
f_info[2] : sentence number
f_info[3] : extension
"""

print("===extract features to .npy===")
for gender in ["male", "female"]:
    SUBDIR_PATH = DIR_PATH / gender
    for f in SUBDIR_PATH.rglob('*.wav'):
        f_info = re.split('[_.]', f.name)
        if int(f_info[2]) > 30:
            continue

        waveform, sample_rate = torchaudio.load(SUBDIR_PATH / f_info[0] / f.name, normalization=True)

        feature = mfcc(
            waveform,
            sample_rate,
            winfunc = np.hamming,
            numcep = hyper_params["numcep"],
            nfilt = 26,
            nfft = 512,
            preemph = 0.97
        )

        speaker = str(f_info[0])

        if speaker in train_speaker_list:
            f_path = FEATURE_PATH / "train" / f.name[:-4]
        elif speaker in val_speaker_list:
            f_path = FEATURE_PATH / "val" / f.name[:-4]
        elif f_info[0] in eval_speaker_list:
            f_path = FEATURE_PATH / "eval" / f.name[:-4]
        else:
            f_path = FEATURE_PATH / f.name[:-4]

        np.save(f_path, feature)