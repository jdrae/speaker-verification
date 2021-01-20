from tqdm import tqdm
import numpy as np


def fit(device, model, loss_func, opt, train_dl):
    model.train()
    for xb, yb in train_dl:
        #flatten
        xb = np.squeeze(xb, axis=1) # rm 1
        xb = np.reshape(xb, (-1, xb.shape[1] * xb.shape[2]))
        # print("after:",xb.shape)

        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)
        cce_loss = loss_func(output,yb)

        cce_loss.backward()
        opt.step()
        opt.zero_grad()

    return cce_loss