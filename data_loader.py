from pathlib import Path
from torch.utils import data
import numpy as np


def get_utt_list(path):
    return [f.name for f in Path(path).rglob('*.npy')]

def get_label_dic(speakers):
    return {label:idx for idx, label in enumerate(speakers)}

class RSRDataset(data.Dataset):
    def __init__(self, utt_list, base_dir, label_dic ={}, nb_time=0, cut=True, is_test=False, tta=False, n_window=0):
        self.utt_list = utt_list
        self.nb_time = nb_time # integer, the number of timesteps for each mini-batch
        self.base_dir = base_dir
        self.label_dic = label_dic
        self.cut = cut
        self.is_test = is_test
        self.tta = tta
        self.n_window = n_window
        if self.nb_time == 0: raise ValueError('when adjusting utterance length, "nb_time" should be input')
        if self.n_window == 0 and is_test: raise ValueError('when doing test time augmentation, "n_window" should be input')
        
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        item = self.utt_list[idx]
        X = np.load(self.base_dir / item)
        X = np.transpose(X)
        x_time = X.shape[1]
        if self.cut: # train set
            if x_time > self.nb_time:
                start_idx = np.random.randint(low = 0, high = x_time - self.nb_time)
                X = X[:, start_idx:start_idx + self.nb_time]
            elif x_time < self.nb_time:
                nb_dup = int(self.nb_time / x_time) + 1
                X = np.tile(X, (1, nb_dup))[:, :self.nb_time]
            else:
                X = X

        # test time augmentation
        if self.tta: # val, eval set
            list_X = []
            if x_time > self.nb_time:
                total = self.nb_time * self.n_window - x_time
                if total <= 0:
                    total = x_time
                overlap = int(total / (self.n_window - 1))
                for i in range(self.n_window):
                    if i == 0:
                        list_X.append(X[:,:self.nb_time])
                    elif i < self.n_window - 1:
                        if (i*overlap+self.nb_time) > x_time:
                            continue
                        else:
                            list_X.append(X[:, i*overlap: i*overlap+self.nb_time])
                    else:
                        list_X.append(X[:, -self.nb_time:])
            elif x_time < self.nb_time:
                nb_dup = int(self.nb_time / x_time) + 1
                list_X.append(np.tile(X, (1,nb_dup))[:,:self.nb_time])
            else:
                list_X.append(X)
            return list_X

        if self.is_test: return X

        y = self.label_dic[item.split('_')[0]] #return label index
        return X, y