import torch
import torch.nn as nn
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from models import *
from proposed_OracleNet import *
from image_enhancement import apply_denoising_or_super_resolution  # Import enhancement function

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4
PRINT_RREQ = 150

x_train, x_test = Load_cifar10_data()
train_dataset = DatasetFolder(x_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

CHANNEL = 'Fading'  # Choose AWGN or Fading
IMG_SIZE = [3, 32, 32]
N_channels = 256
kernel_sz = 5
enc_shape = [48, 8, 8]
KSZ = str(kernel_sz) + 'x' + str(kernel_sz) + '_'

DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
DeepJSCC_V.load_state_dict(torch.load('./JSCC_models/DeepJSCC_VLC_'+KSZ+CHANNEL+'_'+str(N_channels)+'_20.pth.tar')['state_dict'])
DeepJSCC_V.eval()

OraNet = OracleNet(enc_shape[0]).cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(OraNet.parameters(), lr=LEARNING_RATE)

loss_history = []  # Store loss per epoch

if __name__ == '__main__':
    bestLoss = 1e3
    for epoch in range(EPOCHS):
        OraNet.train()
        epoch_loss = 0
        for i, x_input in enumerate(train_loader):
            SNR = torch.randint(0, 28, (x_input.shape[0], 1)).cuda()
            CR = 0.1 + 0.9 * torch.rand(x_input.shape[0], 1).cuda()

            x_input = torch.Tensor(x_input).cuda()
            x_rec = DeepJSCC_V(x_input, SNR, CR, CHANNEL)
            z = DeepJSCC_V.encoder(x_input, SNR)

            x_input = Img_transform(x_input)
            x_rec = Img_transform(x_rec)
            psnr_batch = Compute_IMG_PSNR(x_input, x_rec)
            psnr_batch = torch.Tensor(psnr_batch).cuda()

            z = z.view(-1, enc_shape[0], 8, 8)
            psnr_pred = OraNet(z, SNR, CR)

            for j in range(len(psnr_pred)):
                if psnr_pred[j] < 25:
                    x_rec[j] = apply_denoising_or_super_resolution(x_rec[j])
                elif psnr_pred[j] > 30:
                    psnr_pred[j] = psnr_pred[j] * 1.02

            loss = criterion(psnr_batch, psnr_pred).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % PRINT_RREQ == 0:
                print('Epoch: [{0}][{1}/{2}]	 Loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=loss.item()))

        loss_history.append(epoch_loss / len(train_loader))
        
        df = pd.DataFrame({'Epoch': list(range(1, len(loss_history) + 1)), 'Loss': loss_history})
        df.to_csv('loss_history.csv', index=False)

        OraNet.eval()
        totalLoss = 0
        with torch.no_grad():
            for i, test_input in enumerate(test_loader):
                SNR_TEST = torch.randint(0, 28, (test_input.shape[0], 1)).cuda()
                CR = 0.1 + 0.9 * torch.rand(test_input.shape[0], 1).cuda()

                test_input = torch.Tensor(test_input).cuda()
                test_rec = DeepJSCC_V(test_input, SNR_TEST, CR, CHANNEL)
                z = DeepJSCC_V.encoder(test_input, SNR_TEST)

                test_input = Img_transform(test_input)
                test_rec = Img_transform(test_rec)
                psnr_batch = Compute_IMG_PSNR(test_input, test_rec)
                psnr_batch = torch.Tensor(psnr_batch).cuda()

                z = z.view(-1, enc_shape[0], 8, 8)
                psnr_pred = OraNet(z, SNR_TEST, CR)

                for j in range(len(psnr_pred)):
                    if psnr_pred[j] < 25:
                        test_rec[j] = apply_denoising_or_super_resolution(test_rec[j])
                    elif psnr_pred[j] > 30:
                        psnr_pred[j] = psnr_pred[j] * 1.02

                totalLoss += criterion(psnr_batch, psnr_pred).item() * psnr_batch.size(0)

            averageLoss = totalLoss / len(test_dataset)
            print('averageLoss=', averageLoss)
            if averageLoss < bestLoss:
                torch.save({'state_dict': OraNet.state_dict()}, './JSCC_models/POracleNet_'+CHANNEL+'_Res.pth.tar')
                print('Model saved')
                bestLoss = averageLoss

    # Plot Loss vs Epoch
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()
