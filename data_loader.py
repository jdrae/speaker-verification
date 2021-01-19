from pathlib import Path
from torch.utils import data
import numpy as np

from speaker_list import *
from path_list import *

def get_utt_list(path):
    return [f.name for f in Path(path).rglob('*.npy')]

def get_label_dic(speakers):
    return {label:idx for idx, label in enumerate(speakers)}

class RSRDataLoader(data.Dataset):
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

    def __get_item__(self, idx):
        item = self.utt_list[idx]
        X = np.load(self.base_dir / item)
        if self.cut:
            nb_time = X.shape[1]
            if nb_time > self.nb_time:
                start_idx = np.random.randint(low = 0, high = nb_time - self.nb_time)
                X = X[:, start_idx:start_idx + self.nb_time]
            elif nb_time < self.nb_time:
                nb_dup = int(self.nb_time / nb_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :self.nb_time]
            else:
                X = X

        if not self.return_label:
            return X

        y = self.label_dic[item.split('_')[0]]
        return X, y

dev_utt = get_utt_list(TRAIN_DATA_PATH)
dev_utt = dev_utt[:10] # for test
# print(dev_utt)
dev_labels = get_label_dic(train_speaker_list)
# print(dev_labels)
devset = RSRDataLoader(utt_list=dev_utt, label_dic=dev_labels, base_dir=TRAIN_DATA_PATH, nb_time=2) # nb_time=59049
devset.__get_item__(0)
devset_gen = data.DataLoader(devset, batch_size=10, shuffle=True, drop_last=True, num_workers=3) # batch_size=120, num_workers=12