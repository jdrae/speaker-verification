from tqdm import tqdm
import numpy as np


def fit(epochs, device, model, loss_func, opt, train_dl, log_path):
    loss_log = open(log_path, 'a', buffering=1)
    for epoch in tqdm(range(epochs)):
        model.train()
        for xb, yb in train_dl:
            #flatten
            xb = np.squeeze(xb, axis=1) # rm 1
            xb = np.reshape(xb, (-1, xb.shape[1] * xb.shape[2]))
            # print("after:",xb.shape)

            xb = xb.cuda()
            yb = yb.cuda()

            output = model(xb)
            cce_loss = loss_func(output,yb)

            cce_loss.backward()
            opt.step()
            opt.zero_grad()

        print('Epoch:%d, cce:%.3f\n'%(epoch, cce_loss))
        loss_log.write('Epoch:%d, cce:%.3f\n'%(epoch, cce_loss))
    
    loss_log.close()