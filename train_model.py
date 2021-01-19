from tqdm import tqdm
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl):
    for epoch in tqdm(range(epochs)):
        model.train()
        for xb, yb in train_dl:
            #flatten
            xb = np.squeeze(xb, axis=1) # rm 1
            xb = np.reshape(xb, (-1, xb.shape[1] * xb.shape[2]))
            # print("after:",xb.shape)

            output = model(xb)
            cce_loss = loss_func(output,yb)

            cce_loss.backward()
            opt.step()
            opt.zero_grad()

        print('Epoch%d, cce:%.3f'%(epoch, cce_loss))