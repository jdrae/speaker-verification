# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key:
experiment = Experiment(
    api_key="inzhxYkHljXyQK3HWxaixKNnt",
    project_name="speaker-attention",
    workspace="jdrae",
)

from hyper_params import hyper_params
experiment.log_parameters(hyper_params)

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
print(device)
"""
load data
"""
train_utt = get_utt_list(TRAIN_DATA_PATH)
train_utt = train_utt[:100] # for test
train_labels = get_label_dic(train_speaker_list)

train_ds = RSRDataset(utt_list=train_utt, label_dic=train_labels, base_dir=TRAIN_DATA_PATH, nb_time=hyper_params["nb_time"]) # nb_time=59049
train_ds_gen = data.DataLoader(train_ds, batch_size=hyper_params["batch_size"], shuffle=True) # batch_size=120, num_workers=12

"""
log set
"""
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
log_path = LOG_PATH / "210120-1536.txt"


"""
train
"""
epochs = hyper_params["num_epochs"]
flat_shape = hyper_params["n_mels"] * hyper_params["nb_time"] # 40 is n_mels
label_shape = len(train_speaker_list)

model = Logistic(flat_shape, label_shape).to(device)

opt = torch.optim.Adam(
            model.parameters(),
			lr = hyper_params["lr"],
			weight_decay = hyper_params["weight_decay"],
			amsgrad = 1)

loss_func = torch.nn.CrossEntropyLoss()

log = open(log_path, 'a', buffering=1)

with experiment.train():
	for epoch in tqdm(range(epochs)):
		cce_loss = fit(device, model, loss_func, opt, train_ds_gen)
		experiment.log_metric("cce", cce_loss, step=epoch)
		log.write('Epoch:%d, cce:%.3f\n'%(epoch, cce_loss))

log.close()
