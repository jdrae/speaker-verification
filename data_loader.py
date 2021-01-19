from pathlib import Path
from torch.utils import data
import numpy as np

from speaker_list import *
from path_list import *

def get_utt_list(path):
    return [f.name for f in Path(path).rglob('*.npy')]

def get_label_dic(speakers):
    return {label:idx for idx, label in enumerate(speakers)}

class RSRDataset(data.Dataset):
    def __init__(self, utt_list, label_dic, base_dir, nb_time=0, cut=True, return_label=True):
        self.utt_list = utt_list
        self.nb_time = nb_time # integer, the number of timesteps for each mini-batch
        self.base_dir = base_dir
        self.label_dic = label_dic
        self.cut = cut
        self.return_label = return_label
        if self.cut and self.nb_time == 0: raise ValueError('when adjusting utterance length, "nb_time" should be input')
        
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        item = self.utt_list[idx]
        X = np.load(self.base_dir / item)
        # print("=============")
        # print(self.base_dir / self.utt_list[idx])
        # print("before:", X.shape)
        
        if self.cut:
            nb_time = X.shape[2]
            if nb_time > self.nb_time:
                start_idx = np.random.randint(low = 0, high = nb_time - self.nb_time)
                X = X[:, :, start_idx:start_idx + self.nb_time]
            elif nb_time < self.nb_time:
                nb_dup = int(self.nb_time / nb_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :, :self.nb_time]
            else:
                X = X

        if not self.return_label:
            return X

        # print("after:", X.shape)
        y = self.label_dic[item.split('_')[0]]
        return X, y