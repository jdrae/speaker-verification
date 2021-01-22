from pathlib import Path
from torch.utils import data
import numpy as np


def get_utt_list(path):
    return [f.name for f in Path(path).rglob('*.npy')]

def get_label_dic(speakers):
    return {label:idx for idx, label in enumerate(speakers)}

class RSRDataset(data.Dataset):
    def __init__(self, utt_list, base_dir, label_dic ={}, nb_time=0, cut=True, is_test=False, window_size=0):
        self.utt_list = utt_list
        self.nb_time = nb_time # integer, the number of timesteps for each mini-batch
        self.base_dir = base_dir
        self.label_dic = label_dic
        self.cut = cut
        self.is_test = is_test
        if self.nb_time == 0: raise ValueError('when adjusting utterance length, "nb_time" should be input')
        
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        item = self.utt_list[idx]
        X = np.load(self.base_dir / item)
        
        if self.cut: # train set
            nb_time = X.shape[2]
            if nb_time > self.nb_time:
                start_idx = np.random.randint(low = 0, high = nb_time - self.nb_time)
                X = X[:, :, start_idx:start_idx + self.nb_time]
            elif nb_time < self.nb_time:
                nb_dup = int(self.nb_time / nb_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :, :self.nb_time]
            else:
                X = X

        # test time augmentation
        if self.is_test: # val, eval set
            list_X = []
            nb_time = X.shape[2]
            if nb_time> self.nb_time:
                window_size = int(self.nb_time / 3)
                step = self.nb_time - window_size
                iteration = int( (nb_time - window_size) / step) +1
                for i in range(iteration):
                    if i == 0:
                        list_X.append(X[:, :, :self.nb_time])
                    elif i < iteration - 1:
                        list_X.append(X[:,:, i*step : i*step + self.nb_time])
                    else:
                        list_X. append(X[:,:, -self.nb_time:])
            elif nb_time < self.nb_time:
                nb_dup = int(self.nb_time / nb_time) + 1
                list_X.append(np.tile(X, (1,nb_dup))[:,:,:self.nb_time])
            else:
                list_X.append(X)

            return list_X

        # print("after:", X.shape)
        y = self.label_dic[item.split('_')[0]]
        return X, y