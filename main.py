import os
from tqdm import tqdm
import torch
from torch.utils import data

from model import Logistic
from train_model import fit
from data_loader import get_utt_list, get_label_dic, RSRDataset
from path_list import TRAIN_DATA_PATH, LOG_PATH
from speaker_list import train_speaker_list

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
load data
"""
train_utt = get_utt_list(TRAIN_DATA_PATH)
train_utt = train_utt[:100] # for test
train_labels = get_label_dic(train_speaker_list)

nb_time = 324
train_ds = RSRDataset(utt_list=train_utt, label_dic=train_labels, base_dir=TRAIN_DATA_PATH, nb_time=nb_time) # nb_time=59049
train_ds_gen = data.DataLoader(train_ds, batch_size=100, shuffle=True) # batch_size=120, num_workers=12

"""
log set
"""
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
log_path = LOG_PATH / "210119-1700.txt"


"""
train
"""
epochs = 20
flat_shape = 40 * nb_time # 40 is n_mels
label_shape = len(train_speaker_list)

model = Logistic(flat_shape, label_shape)
# model.to(device)
opt = torch.optim.Adam(
            model.parameters(),
			lr = 0.001,
			weight_decay = 0.001,
			amsgrad = 1)

loss_func = torch.nn.CrossEntropyLoss()
fit(epochs, model, loss_func, opt, train_ds_gen, log_path)