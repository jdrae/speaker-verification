from hyper_params import hyper_params as hp

if hp["comet"]:
	# import comet_ml at the top of your file
	from comet_ml import Experiment

	# Create an experiment with your api key:
	experiment = Experiment(
		api_key="inzhxYkHljXyQK3HWxaixKNnt",
		project_name="speaker-attention",
		workspace="jdrae",
	)

	experiment.log_parameters(hp)

import os
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np

from model import Logistic
from train_model import fit, test
from data_loader import get_utt_list, get_label_dic, RSRDataset
from config.path_list import *
from config.speaker_list import train_speaker_list
from transformer.transformer import Transformer
from transformer.encoder import Encoder
from transformer.pooling import SelfAttentionPooling

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
print("read trials")
with open(DEV_VAL_TRIALS_PATH, 'r') as f: # dev val trials
	li = f.readlines()
dev_val_trial = []
for line in li:
	dev_val_trial.append(tuple(line.strip().split('/')))

with open(VAL_TRIALS_PATH, 'r') as f: # val trials
	li = f.readlines()
val_trial = []
for line in li:
	val_trial.append(tuple(line.strip().split('/')))
with open(VAL_PWD_PATH, 'r') as f: # val pwd
	li = f.readlines()
val_pwd = []
for line in li:
	val_pwd.append(tuple(line.strip().split('/')))

with open(EVAL_TRIALS_PATH, 'r') as f: # eval trials
	li = f.readlines()
eval_trial = []
for line in li:
	eval_trial.append(tuple(line.strip().split('/')))
with open(EVAL_PWD_PATH, 'r') as f: # eval pwd
	li = f.readlines()
eval_pwd = []
for line in li:
	eval_pwd.append(tuple(line.strip().split('/')))
print("got trials")

# parameters
epochs = hp["num_epochs"]
if hp["dev"]:
	epochs = 1
	train_utt = train_utt[:1000]
	# val_trial = dev_val_trial
val_tral = val_trial[:411310]
eval_tral = eval_trial[:377471]

val_tta = False
if val_tta:
	val_bs = 1
	val_cut = False
else:
	val_bs = 64
	val_cut = True

# dataloader
train_ds 		= RSRDataset(
					utt_list=train_utt, 
					label_dic=train_labels, 
					base_dir=TRAIN_DATA_PATH,
					nb_time=hp["nb_time"]
				)
train_ds_gen	= data.DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, drop_last = True, num_workers=hp["num_workers"])
val_ds 			= RSRDataset(
					utt_list=val_utt, 
					base_dir=VAL_DATA_PATH, 
					is_test=True, # doesn't return label
					cut = val_cut, # do time augmentation instead of cutting
					tta=val_tta,
					nb_time=hp["nb_time"],
					n_window = hp["n_window"]
				)
val_ds_gen 		= data.DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=hp["num_workers"]) # batch size should be 1?
eval_ds 		= RSRDataset(
					utt_list=eval_utt, 
					base_dir=EVAL_DATA_PATH, 
					is_test=True, # doesn't return label
					cut=False, # do time augmentation instead of cutting
					tta=True,
					nb_time=hp["nb_time"],
					n_window = hp["n_window"]
				)
eval_ds_gen 	= data.DataLoader(eval_ds, batch_size=1, shuffle=False,num_workers=hp["num_workers"]) # batch size should be 1?


"""
log set
"""
from datetime import datetime
from pytz import timezone
now = datetime.now(timezone('Asia/Seoul'))
f_name = now.strftime("%m%d-%H%M%S") + ".txt"
r_name = now.strftime("%m%d-%H%M%S") + "-result.txt"
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
eer_path = LOG_PATH / f_name
res_path = LOG_PATH / r_name

"""
train
"""
# flat_shape = hp["numcep"] * hp["nb_time"]
d_input = hp["numcep"]
label_shape = len(train_speaker_list)

# model
d_m = hp["d_m"]
encoder = Encoder(
			d_input = d_input,
			n_layers = 2,
			d_k = d_m,
			d_v = d_m,
			d_m = d_m,
			d_ff = hp["d_ff"],
			dropout = 0.1
		).to(device)
pooling = SelfAttentionPooling(d_m, dropout=0.1).to(device)
model = Transformer(encoder, pooling, d_m, label_shape, dropout=0.2).to(device)

opt = torch.optim.Adam(
            model.parameters(),
			lr = hp["lr"],
			weight_decay = hp["weight_decay"]
			)

loss_func = torch.nn.CrossEntropyLoss()

best_eer = 99.
if hp["comet"]:
	with experiment.train():
		for epoch in tqdm(range(epochs)):
			cce_loss = fit(model, loss_func, opt, train_ds_gen, device)
			experiment.log_metric("cce", cce_loss, epoch=epoch)
			
			val_eer = test(model, val_ds_gen, val_utt, val_pwd, val_trial, device, tta=val_tta)
			experiment.log_metric("val eer", val_eer, epoch=epoch)
			if float(val_eer) < best_eer:
				print("New best EER: %f"%float(val_eer))
				best_eer = float(val_eer)

	with experiment.test():
		eval_eer = test(model, eval_ds_gen, eval_utt, eval_pwd, eval_trial, device, tta=True)
		experiment.log_metric("eval eer", eval_eer, epoch=1)
elif hp["log"]:
	f_eer = open(eer_path, "a+", buffering =1)
	f_res = open(res_path, "a+", buffering =1)
	f_eer.write("Epoch\ttrain_cce\tval_eer\n")
	for epoch in tqdm(range(epochs), desc='epoch'):
		cce_loss = fit(model, loss_func, opt, train_ds_gen, device)
		val_eer = test(model, val_ds_gen, val_utt, val_pwd, val_trial, device, tta=val_tta)
		print("epoch:",epoch)
		print("train_cce:", cce_loss)
		print("val_eer:%.3f"%(val_eer))
		if float(val_eer) < best_eer:
			print("New best EER: %f"%float(val_eer))
			best_eer = float(val_eer)
		f_eer.write("%d\t%.4f\t%.4f\n"%(epoch, cce_loss, val_eer))
	eval_eer = test(model, eval_ds_gen, eval_utt, eval_pwd, eval_trial, device, tta=True)
	print("eval_eer:%.3f"%(eval_eer))
	f_res.write("Eval eer:%.3f\n"%(eval_eer))
	f_res.write("Best eer:%.3f\n"%(best_eer))
	f_eer.close()
	f_res.close()
else:
	for epoch in tqdm(range(epochs), desc='epoch'):
		cce_loss = fit(model, loss_func, opt, train_ds_gen, device)
		val_eer = test(model, val_ds_gen, val_utt, val_pwd, val_trial, device, tta=val_tta)
		print("epoch:",epoch)
		print("train_cce:", cce_loss)
		print("val_eer:%.3f"%(val_eer))
		if float(val_eer) < best_eer:
			print("New best EER: %f"%float(val_eer))
			best_eer = float(val_eer)
	eval_eer = test(model, eval_ds_gen, eval_utt, eval_pwd, eval_trial, device, tta=True)
	print("eval_eer:%.3f"%(eval_eer))
