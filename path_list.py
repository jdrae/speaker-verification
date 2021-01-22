from pathlib import Path

DATA_PATH       = Path("/workspace/user/data")
DIR_PATH       = DATA_PATH / "RSR2015/sph"
FEATURE_PATH    = DATA_PATH / "melspectrogram"
TRIALS_PATH     = DATA_PATH / "custom_trials"

SPEAKER_INFO_PATH = DATA_PATH / "RSR2015/infos/spkrinfo.lst"

VAL_TRIALS_PATH = TRIALS_PATH / "val_trials.txt"
EVAL_TRIALS_PATH = TRIALS_PATH / "eval_trials.txt"

VAL_PWD_PATH = TRIALS_PATH / "val_pwd.txt"
EVAL_PWD_PATH = TRIALS_PATH / "eval_pwd.txt"

TRAIN_SPEAKER_PATH = TRIALS_PATH / "train_speaker.txt"
VAL_SPEAKER_PATH = TRIALS_PATH / "val_speaker.txt"
EVAL_SPEAKER_PATH = TRIALS_PATH / "eval_speaker.txt"

TRAIN_DATA_PATH = FEATURE_PATH / "train"
VAL_DATA_PATH = FEATURE_PATH / "val"
EVAL_DATA_PATH = FEATURE_PATH / "eval"

LOG_PATH = DATA_PATH / "log"