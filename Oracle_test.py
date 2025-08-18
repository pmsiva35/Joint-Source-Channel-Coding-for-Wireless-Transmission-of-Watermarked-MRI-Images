import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from models import *
from OracleNet import OracleNet

# GPU Memory Cleanup
torch.cuda.empty_cache()

# Parameters
CHANNEL = 'AWGN'  # Choose AWGN or Fading
CR_INDEX = torch.Tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).int()
IMG_SIZE = [3, 32, 32]  # Ignored, since we use Kodak images
N_channels = 256
kernel_sz = 5
enc_shape = [48, 8, 8]
KSZ = f"{kernel_sz}x{kernel_sz}_"

# Load a Single Kodak Image
img_path = './data/Kodak24/kodim13.png'
img = Image.open(img_path).convert('RGB')

# Convert Image to Tensor
img = np.array(img).astype('float32') / 255
img = np.transpose(img, (2, 0, 1))  # Change to (C, H, W)
img = torch.Tensor(img).unsqueeze(0).cuda()

# Load Models
DeepJSCC_V = ADJSCC_V(enc_shape, kernel_sz, N_channels).cuda()
DeepJSCC_V.load_state_dict(torch.load(f'./JSCC_models/DeepJSCC_VLC_{KSZ}{CHANNEL}_{N_channels}_20.pth.tar')['state_dict'])
DeepJSCC_V.eval()

OraNet = OracleNet(enc_shape[0]).cuda()
OraNet.load_state_dict(torch.load(f'./JSCC_models/OracleNet_{CHANNEL}_Res.pth.tar')['state_dict'])
OraNet.eval()

criterion = nn.MSELoss().cuda()
MSE_pred = np.zeros((10, 10))

# Model Evaluation
for m in range(10):
    cr = 1 / CR_INDEX[m]
    for k in range(10):
        SNR_TEST = 3*(k-1)*torch.ones((img.shape[0], 1, 1, 1)).cuda()
        CR = cr * torch.ones((1, 1)).cuda()

        with torch.no_grad():
            test_rec = DeepJSCC_V(img, SNR_TEST, CR, CHANNEL)
            z = DeepJSCC_V.encoder(img, SNR_TEST)

            img0 = Img_transform(img)
            img_rec = Img_transform(test_rec)
            psnr = Compute_IMG_PSNR(img0, img_rec)

            z = z.view(-1, enc_shape[0], 8, 8)
            psnr_pred = OraNet(z, SNR_TEST, CR)

            mse_loss = criterion(torch.Tensor([psnr]).cuda(), psnr_pred).item()
            print(f'CR = {cr.item():.3f}, SNR = {3 * (k - 1):.1f} dB, PSNR = {psnr:.2f}, Predicted = {psnr_pred.item():.2f}, MSE = {mse_loss:.5f}')
            
            MSE_pred[m, k] = mse_loss

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img0[0].cpu().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img_rec[0].cpu().numpy().transpose(1, 2, 0))
plt.axis('off')
plt.title("Reconstructed Image")

plt.show()
