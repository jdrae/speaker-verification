from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def flatten(xb):
    xb = np.squeeze(xb, axis=1) # rm 1
    xb = np.reshape(xb, (-1, xb.shape[-2] * xb.shape[-1]))
    return xb


def cos_sim(a, b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))



def fit(model, loss_func, opt, ds_gen, device):
    model.train()
    for xb, yb in tqdm(ds_gen, total=len(ds_gen)):
        xb = xb.to(device) # (64, 13, 200) convolution in encoder
        yb = yb.to(device)
        output = model(xb.float()) # (64, 196)
        cce_loss = loss_func(output,yb)
        opt.zero_grad()
        cce_loss.backward()
        opt.step()
    return cce_loss

# test time augmented
def test(mode, model, ds_gen, utt_list, pwd_list, trial_list, epoch, device):
    if mode not in ['val', 'eval']: raise ValueError('mode should be either "val" or "eval"')
    model.eval()
    with torch.no_grad():
        """extract utterance embeddings"""
        utt_emb_l = [] # (num_utt, num_feature)

        if mode =='val':
            for xb in tqdm(ds_gen, total=len(ds_gen)):
                xb = xb.to(device) #(64, 13, 200)
                output = model(xb.float(),is_test=True) # (64, 220)
                utt_emb_l.extend(output.cpu().numpy())
        if mode =='eval': # tta
            for m_batch in tqdm(ds_gen, total=len(ds_gen)): # len(m_batch) = n_window (almost)
                output_l = []
                for xb in m_batch:
                    xb = xb.to(device) #(1, 13, 200)
                    
                    output = model(xb.float(),is_test=True) # (1, 220)
                    output_l.append(output.cpu().numpy())  # (num_output, num_feature)
                # average of tta
                utt_emb_l.append(np.mean(output_l, axis=0))

        """create utterance embeddings dictionary"""
        utt_emb_d = {} # (num_utt, num_feature)
        if not len(utt_list) == len(utt_emb_l): # check
            print(len(utt_list), len(utt_emb_l))
            exit()
        for k, v in zip(utt_list, utt_emb_l):  # ? 순서
            k = k[:-4] # remove extension .npy
            utt_emb_d[k] = v

        """speaker embeddings"""
        # speaker embedings avg. of utt-emb
        spk_emb_d = {} # (num_speaker, num_feature)
        for line in pwd_list:
            pwd_key, utt1, utt2, utt3 = line.strip().split('/')
            spk_emb_l = [] # (3, num_feature)
            for utt in [utt1, utt2, utt3]:
                spk_emb_l.append(utt_emb_d[utt])
            spk_emb_d[pwd_key] = np.mean(spk_emb_l, axis=0) # (num_feature, )
        if not len(pwd_list) == len(spk_emb_d): # check
            print(len(pwd_list), len(spk_emb_d))
            exit()

        """calculate eer"""
        y_score = [] # score for each sample
        y = [] # label for each sample
        
        for line in trial_list:
            spk, utt, trg = line.strip().split('/')
            y.append(int(trg))
            y_score.append(cos_sim(spk_emb_d[spk], utt_emb_d[utt]))
        # fpr: false positive rates
        # tpr: true positive rates
        fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer