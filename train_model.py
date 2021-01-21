from tqdm import tqdm
import numpy as np
import torch


def fit(model, loss_func, opt, ds_gen, device):
    model.train()
    for xb, yb in ds_gen:
        #flatten
        xb = np.squeeze(xb, axis=1) # rm 1
        xb = np.reshape(xb, (-1, xb.shape[1] * xb.shape[2]))

        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)
        cce_loss = loss_func(output,yb)

        cce_loss.backward()
        opt.step()
        opt.zero_grad()

    return cce_loss

# time augmented
def test(mode, model, ds_gen, utt_list, trial_list, save_dir, epoch, device):
    if mode not in ['val', 'eval']: raise ValueError('mode should be either "val" or "eval"')
    model.eval()
    with torch.set_grad_enabled(False):
        # extract utterance embeddings from tta ds
        utt_emb_l = []
        for m_batch in tqdm(ds_gen, total=len(ds_gen)):
            output_l = []
            for xb in m_batch:
                #flatten
                xb = np.squeeze(xb, axis=1) # rm 1
                xb = np.reshape(xb, (-1, xb.shape[1] * xb.shape[2]))

                xb = xb.to(device)
                output = model(xb)
                output_l.extend(output.cpu().numpy())  #>>> (batchsize, codeDim)
            utt_emb_l.append(np.mean(output_l, axis=0))
        
        utt_emb_d = {}
        if not len(utt_list) == len(utt_emb_l): # check
            print(len(utt_list), len(utt_emb_l))
            exit()
        for k, v in zip(utt_list, utt_emb_l):
            utt_emb_d[k] = v

        print(utt_emb_l[0])
        print(np.shape(utt_emb_l))
        exit()
        # speaker embedings avg. of utt-emb



        # calculate eer
        y_score = [] # score for each sample
        y = [] # label for each sample
        
        for line in trial_list:
            pwd_key, utt, label = line.strip().split('/')
            y.append(pwd_key[:4])
            y_score.append(cos_sim(embedding_d[utt]))