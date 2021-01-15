from pathlib import Path
import re
import torch
import torchaudio
import numpy as np


DATA_PATH       = Path("/workspace/user/data")
DIR_PATH       = DATA_PATH / "RSR2015/sph"
FEATURE_PATH    = DATA_PATH / "melspectrogram"

FEATURE_PATH.mkdir(parents=True, exist_ok=True)

    
"""
f_info[0] : speaker ID
f_info[1] : session number
f_info[2] : sentence number
f_info[3] : extension
"""

print("===extract features to .npy===")
for gender in ["male", "female"]:
    SUBDIR_PATH = DIR_PATH / gender
    for idx, f in enumerate(SUBDIR_PATH.rglob('*.wav')):
        f_info = re.split('[_.]', f.name)
        if int(f_info[2]) > 30:
            continue
        if idx > 30: # for test
            break

        waveform, sample_rate = torchaudio.load(SUBDIR_PATH / f_info[0] / f.name, normalization=True)
        mel_specgram = torchaudio.transforms.MelSpectrogram(
                sample_rate, # 16000
                n_fft=512,
                win_length=int(sample_rate * 0.025),
                hop_length=int(sample_rate * 0.01),
                window_fn=torch.hamming_window,
                n_mels=40
            )(waveform)

        f_path = FEATURE_PATH / f.name[:-4]
        np.save(f_path, mel_specgram)