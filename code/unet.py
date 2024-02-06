####################
# IMPORTING LIBRARY
####################
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
import random
import cv2

##############################
# Crop and Concatenate Class
##############################
class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        return x
#############################
# U-Net Model Architecture
#############################
def MakeUnet(printtoggle=False):

    class Unet(nn.Module):
        def __init__(self,printtoggle):
            super().__init__()
            # print toggle
            self.print = printtoggle
            self.crop_and_concat = CropAndConcat()
            ######################
            # ENCODER LAYERS
            ######################
            # Encoder Layer 1
            self.enc_conv1_a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1_a = nn.BatchNorm2d(64)
            self.enc_conv1_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn1_b = nn.BatchNorm2d(64)
            self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            # Encoder Layer 2
            self.enc_conv2_a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2_a = nn.BatchNorm2d(128)
            self.enc_conv2_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn2_b = nn.BatchNorm2d(128)
            self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            # Encoder Layer 3
            self.enc_conv3_a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3_a = nn.BatchNorm2d(256)
            self.enc_conv3_b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn3_b = nn.BatchNorm2d(256)
            self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            # Encoder Layer 4
            self.enc_conv4_a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.bn4_a = nn.BatchNorm2d(512)
            self.enc_conv4_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.bn4_b = nn.BatchNorm2d(512)
            self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            # Latent Layer 5
            self.enc_conv5_a = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
            self.bn5_a = nn.BatchNorm2d(1024)
            self.enc_conv5_b = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
            self.bn5_b = nn.BatchNorm2d(1024)
            ######################
            # DECODER LAYERS
            ######################
            # Decoder Layer 1
            self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec_conv1_a = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.bn6_a = nn.BatchNorm2d(512)
            self.dec_conv1_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.bn6_b = nn.BatchNorm2d(512)
            # Decoder Layer 2
            self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec_conv2_a = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.bn7_a = nn.BatchNorm2d(256)
            self.dec_conv2_b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn7_b = nn.BatchNorm2d(256)
            # Decoder Layer 3
            self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec_conv3_a = nn.Conv2d(256, 128, kernel_size=3, padding=1)
            self.bn8_a = nn.BatchNorm2d(128)
            self.dec_conv3_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn8_b = nn.BatchNorm2d(128)
            # Decoder Layer 4
            self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec_conv4_a = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.bn9_a = nn.BatchNorm2d(64)
            self.dec_conv4_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn9_b = nn.BatchNorm2d(64)
            # Output Layer
            self.output_conv = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x):
            # Input Padding
            xpad = F.pad(x, pad=(0,0,4,4), mode='constant', value=0)
            if self.print: print(f'Input: {list(x.shape)}')
            ##########
            # Encoder
            ##########
            # Encoder Layer 1
            xe1a = F.leaky_relu(self.bn1_a(self.enc_conv1_a(xpad)))
            xe1b = F.leaky_relu(self.bn1_b(self.enc_conv1_b(xe1a)))
            xep1 = self.max_pool1(xe1b)
            if self.print: print(f'1st encoder block: {list(xep1.shape)}')
            # Encoder Layer 2
            xe2a = F.leaky_relu(self.bn2_a(self.enc_conv2_a(xep1)))
            xe2b = F.leaky_relu(self.bn2_b(self.enc_conv2_b(xe2a)))
            xep2 = self.max_pool2(xe2b)
            if self.print: print(f'2nd encoder block: {list(xep2.shape)}')
            # Encoder Layer 3
            xe3a = F.leaky_relu(self.bn3_a(self.enc_conv3_a(xep2)))
            xe3b = F.leaky_relu(self.bn3_b(self.enc_conv3_b(xe3a)))
            xep3 = self.max_pool3(xe3b)
            if self.print: print(f'3rd encoder block: {list(xep3.shape)}')
            # Encoder Layer 4
            xe4a = F.leaky_relu(self.bn4_a(self.enc_conv4_a(xep3)))
            xe4b = F.leaky_relu(self.bn4_b(self.enc_conv4_b(xe4a)))
            xep4 = self.max_pool4(xe4b)
            if self.print: print(f'4th encoder block: {list(xep4.shape)}')
            # Latent Layer
            xe5a = F.leaky_relu(self.bn5_a(self.enc_conv5_a(xep4)))
            xe5b = F.leaky_relu(self.bn5_b(self.enc_conv5_b(xe5a)))
            if self.print: print(f'Latent block: {list(xe5b.shape)}')
            ##################
            # Decoder Layers
            ##################
            # Decoder Layer 1
            xdu1 = self.up_conv1(xe5b)
            xduc1 = self.crop_and_concat(xdu1, xe4b)
            xd1a = F.leaky_relu(self.bn6_a(self.dec_conv1_a(xduc1)))
            xd1b = F.leaky_relu(self.bn6_b(self.dec_conv1_b(xd1a)))
            if self.print: print(f'1st Decoder block: {list(xd1b.shape)}')
            # Decoder Layer 2
            xdu2 = self.up_conv2(xd1b)
            xduc2 = self.crop_and_concat(xdu2, xe3b)
            # xduc2 = torch.cat([xdu2, xe3b], dim=1)
            xd2a = F.leaky_relu(self.bn7_a(self.dec_conv2_a(xduc2)))
            xd2b = F.leaky_relu(self.bn7_b(self.dec_conv2_b(xd2a)))
            if self.print: print(f'2nd Decoder block: {list(xd2b.shape)}')
            # Decoder Layer 3
            xdu3 = self.up_conv3(xd2b)
            xduc3 = self.crop_and_concat(xdu3, xe2b)
            # xduc3 = torch.cat([xdu3, xe2b], dim=1)
            xd3a = F.leaky_relu(self.bn8_a(self.dec_conv3_a(xduc3)))
            xd3b = F.leaky_relu(self.bn8_b(self.dec_conv3_b(xd3a)))
            if self.print: print(f'3rd Decoder block: {list(xd3b.shape)}')
            # Decoder Layer 4
            xdu4 = self.up_conv4(xd3b)
            xduc4 = self.crop_and_concat(xdu4, xe1b)
            # xduc4 = torch.cat([xdu4, xe1b], dim=1)
            xd4a = F.leaky_relu(self.bn9_a(self.dec_conv4_a(xduc4)))
            xd4b = F.leaky_relu(self.bn9_b(self.dec_conv4_b(xd4a)))
            if self.print: print(f'4th Decoder block: {list(xd4b.shape)}')
            # Output Layer
            x_out = self.output_conv(xd4b)
            if self.print: print(f'Output Layer: {list(x_out.shape)}')
            return x_out
    # create the model instance
    net = Unet(printtoggle)
    # loss function
    lossfun = nn.BCEWithLogitsLoss() # lossfun = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=.0001,weight_decay=1e-5)
    return net, lossfun, optimizer

#####################
# Loading the model
#####################
def get_prediction_image(image, model, checkpoint, net,device):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Device Available is : {device}")

    # checkpoint = torch.load('/content/drive/MyDrive/Deep Learning for Unet/Saved Models/my_checkpoint_rutwik_training2.pth.tar')
    # model = net
    # model.load_state_dict(checkpoint['state_dict'])
    # model = model.to(device)

    model.eval()

    # image = Image.open('/home/pear/AerialRobotics/Aerial/Unet_sim2real/img14.png')
    # image = image.rotate(-90, expand=True)
    # plt.imshow(image)

    preprocess = transforms.Compose([
        transforms.Resize((360, 480)),
        # transforms.RandomRotation(180),
        # transforms.CenterCrop((640,360)),
        transforms.ToTensor(),
        transforms.Normalize([0.3353, 0.2599, 0.2292], [0.2137,0.1890,0.1884]),
    ])
    
    input_tensor = preprocess(image)
    torchvision.utils.save_image(input_tensor, f"Image_going_into_network.png")
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)

    predictions = torch.nn.functional.sigmoid(output)
    predictions = (predictions > 0.5).float()
    predictions = predictions.cpu().numpy()[0]
    pred = np.squeeze(predictions)
    # pred_np = np.reshape(predictions, (368,480))
    return pred 
    # torchvision.utils.save_image(predictions, f"prediction_image.png")
        
def imshow_gray(img):
    img = img.numpy()
    plt.imshow(img, cmap='gray')
    plt.show()




