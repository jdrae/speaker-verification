from pathlib import Path

DATA_PATH       = Path("/workspace/user/data")
DIR_PATH       = DATA_PATH / "RSR2015/sph"
FEATURE_PATH    = DATA_PATH / "melspectrogram"

SPEAKER_INFO_PATH = DATA_PATH / "RSR2015/infos/spkrinfo.lst"

VAL_TRIALS_PATH = DATA_PATH / "custom_trials/val_trials.txt"
EVAL_TRIALS_PATH = DATA_PATH / "custom_trials/eval_trials.txt"


VAL_SPEAKER_PATH = DATA_PATH / "custom_trials/val_speaker.txt"
EVAL_SPEAKER_PATH = DATA_PATH / "custom_trials/eval_speaker.txt"

TRAIN_SPEAKER_PATH = DATA_PATH / "custom_trials/train_speaker.txt"