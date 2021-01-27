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
import numpy as np

from model import Logistic
from train_model import fit, test
from data_loader import get_utt_list, get_label_dic, RSRDataset
from path_list import *
from speaker_list import train_speaker_list

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
"""
load data
"""
# file list
train_labels = get_label_dic(train_speaker_list)
train_utt = get_utt_list(TRAIN_DATA_PATH)
val_utt = get_utt_list(VAL_DATA_PATH)
eval_utt = get_utt_list(EVAL_DATA_PATH)

# get trial
with open(VAL_TRIALS_PATH , 'r') as f:
	val_trial = f.readlines()
# with open(DEV_VAL_TRIALS_PATH , 'r') as f: # test
# 	dev_val_trial = f.readlines()
with open(VAL_PWD_PATH , 'r') as f:
	val_pwd = f.readlines()
with open(EVAL_TRIALS_PATH, 'r') as f:
	eval_trial = f.readlines()
with open(EVAL_PWD_PATH , 'r') as f:
	eval_pwd = f.readlines()

# for debugging
train_utt = train_utt
val_utt = val_utt
# val_trial = dev_val_trial
val_trial = val_trial
eval_utt = eval_utt
eval_trial = eval_trial

# dataloader
train_ds 		= RSRDataset(
					utt_list=train_utt, 
					label_dic=train_labels, 
					base_dir=TRAIN_DATA_PATH,
					nb_time=hyper_params["nb_time"]
				)
train_ds_gen	= data.DataLoader(train_ds, batch_size=hyper_params["batch_size"], shuffle=True)
val_ds 			= RSRDataset(
					utt_list=val_utt, 
					base_dir=VAL_DATA_PATH, 
					is_test=True, # doesn't return label
					cut=False, # do time augmentation instead of cutting
					nb_time=hyper_params["nb_time"],
					n_window = hyper_params["n_window"]
				)
val_ds_gen 		= data.DataLoader(val_ds, batch_size=1, shuffle=False) # batch size should be 1?
eval_ds 		= RSRDataset(
					utt_list=eval_utt, 
					base_dir=EVAL_DATA_PATH, 
					is_test=True, # doesn't return label
					cut=False, # do time augmentation instead of cutting
					nb_time=hyper_params["nb_time"],
					n_window = hyper_params["n_window"]
				)
eval_ds_gen 	= data.DataLoader(eval_ds, batch_size=1, shuffle=False) # batch size should be 1?


"""
log set
"""
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
if not os.path.exists(LOG_PATH  / 'results/'):
	os.makedirs(LOG_PATH / 'results/')
if not os.path.exists(LOG_PATH  / 'models/'):
	os.makedirs(LOG_PATH / 'models/')
# log_path = LOG_PATH / "210120-1536.txt"


"""
train
"""
epochs = hyper_params["num_epochs"]
flat_shape = hyper_params["n_mels"] * hyper_params["nb_time"]
label_shape = len(train_speaker_list)

model = Logistic(flat_shape, label_shape).to(device)

opt = torch.optim.Adam(
            model.parameters(),
			lr = hyper_params["lr"]
			)

loss_func = torch.nn.CrossEntropyLoss()

# best_eer = 99.
# for epoch in tqdm(range(epochs), desc='epoch'):
# 	cce_loss = fit(model, loss_func, opt, train_ds_gen, device)
# 	val_eer = test("val", model, val_ds_gen, val_utt, val_pwd, val_trial, epoch, device)
# 	print("epoch:",epoch)
# 	print("train_cce:", cce_loss)
# 	print("val_eer:%.3f"%(val_eer))
# 	if float(val_eer) < best_eer:
# 		print("New best EER: %f"%float(val_eer))
# 		best_eer = float(val_eer)
# eval_eer = test("eval", model, eval_ds_gen, eval_utt, eval_pwd, eval_trial, 1, device)
# print("eval_eer:%.3f"%(eval_eer))

best_eer = 99.
with experiment.train():
	for epoch in tqdm(range(epochs)):
		cce_loss = fit(model, loss_func, opt, train_ds_gen, device)
		experiment.log_metric("cce", cce_loss, epoch=epoch)
		
		val_eer = test("val", model, val_ds_gen, val_utt, val_pwd, val_trial, epoch, device)
		experiment.log_metric("val eer", val_eer, epoch=epoch)
		if float(val_eer) < best_eer:
			print("New best EER: %f"%float(val_eer))
			best_eer = float(val_eer)

with experiment.test():
	eval_eer = test("eval", model, eval_ds_gen, eval_utt, eval_pwd, eval_trial, 1, device)
	experiment.log_metric("eval eer", eval_eer, epoch=1)