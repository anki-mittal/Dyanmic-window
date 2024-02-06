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

class CustomDataset(Dataset):
    def __init__(self, img_folder, mask_folder, img_transform=None, mask_transform=None):
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_files_list = sorted([f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
        self.mask_files_list = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, f"img_{str(idx+1).zfill(4)}.png")
        mask_name = os.path.join(self.mask_folder, f"Image{str(idx+1).zfill(4)}.png")

        # The Input Image is RGB
        image = Image.open(img_name).convert("RGB")
        # The source Image is Grayscale.
        mask = Image.open(mask_name).convert("L")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

#############################
# U-Net Model Architecture
#############################
# Crop and Concatenate Class
class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        return x

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
            if self.print: print(f'2nd Decoder block: {list(xd1b.shape)}')
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

##########################
# FUNCTIONS FOR TRAINING
##########################
# Save Checkpoints
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
# Load Checkpoints
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
# Check Accuracy
def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for X,y in loader:
            y = F.pad(y, (0,0,4,4), mode='constant', value=0)
            X = X.to(device)
            y = y.to(device)
            yHat = torch.sigmoid(model(X))
            yHat = (yHat > .5).float()
            num_correct += (yHat == y).sum()
            num_pixels += torch.numel(yHat) # ??
            dice_score += (2 * (yHat * y).sum()) / ((yHat + y).sum() + 1e-8)
            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
            print(f"Dice score: {dice_score/len(loader)}")
            model.train()
# Save Predictions
def save_predictions_as_images(loader, model, folder, device):
    model.eval()
    for ii, (X, y) in enumerate(loader):
        y = F.pad(y, (0,0,4,4), mode='constant', value=0)
        X = X.to(device=device)
        with torch.no_grad():
            yHat = torch.sigmoid(model(X))
            yHat = (yHat > 0.5).float()
        torchvision.utils.save_image(yHat, f"/{folder}_prediction_{ii}.png")
        torchvision.utils.save_image(y, f"/{folder}_ground_truth_{ii}.png")
    model.train()
# Train Model
def train_unet_model(loader, model, optimizer, lossfun, scaler, device):
    # for tracking progress
    loop = tqdm(loader)
    for (batch_index), (data, targets) in enumerate(loop):
        targets = F.pad(targets, (0,0,4,4), mode='constant', value=0)
        # pushing
        data = data.to(device)
        targets = targets.to(device)
        # Forward pass and computing loss.
        with torch.cuda.amp.autocast():
            yHat = model(data)
            loss = lossfun(yHat, targets)
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Update the tqdm loop.
        loop.set_postfix(loss=loss.item())


def main():
    # Define your transformations
    img_transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the datasets
    dataset = CustomDataset("/home/ankit/Documents/STUDY/RBE595/HW3a/dataset/windowraw",
                            "/home/ankit/Documents/STUDY/RBE595/HW3a/dataset/labelimage",
                            img_transformations,
                            mask_transformations)

    # Split the dataset into training, validation and testing sets
    indices = list(range(len(dataset)-1))
    train_indices, temp_indices = train_test_split(indices, test_size=0.005)  # 80% for training
    valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5)  # 10% for validation, 10% for testing

    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1)  # adjust batch size as needed
    test_loader = DataLoader(test_dataset, batch_size=1)  # adjust batch size as needed

    # Debugging
    # Get a batch of training data
    inputs, masks = next(iter(train_loader))
    input_test, masks_test = next(iter(test_loader))
    input_valid, masks_valid = next(iter(valid_loader))

    print(f"Train Set Input Image Shape (1 batch) : {inputs.shape}")
    print(f"Train Set Image Shape (1 batch) : {masks.shape}")

    print(f"Test Set Input Image Shape: {input_test.shape}")
    print(f"Test Set Label Image Shape: {masks_test.shape}")

    print(f"Validation Set Input Image Shape: {input_valid.shape}")
    print(f"Validation Set Label Image Shape: {masks_valid.shape}")

    print(f"\n Input Image Matrix\n{inputs}")
    print(f"\n Target Image Matrix\n{masks}")

    ######################################
    # Functions for Display Purposes Only
    ######################################
    # def imshow_gray(img):
    #     img = img.numpy()[0]
    #     plt.imshow(img, cmap='gray')
    #     plt.show()

    # def imshow_color(img):
    #     img = img # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # # Showing images in the first batch
    # batch_size = 2
    # for ii in range(batch_size):
    #     print(f"Set {ii+1}, Image v/s Label")
    #     imshow_color(inputs[ii])
    #     imshow_gray(masks[ii])

    # Checking if GPU is Available.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Device Available : {device}")

    ########################
    # RUNNING SANITY CHECK
    ######################## 
    # # test the model with one batch
    # net, lossfun, optimizer = MakeUnet(True)

    # X,y = next(iter(train_loader))
    # y = F.pad(y, (0,0,4,4), mode='constant', value=0)
    # yHat = net(X)

    # prob = torch.sigmoid(yHat)
    # pred = (prob >= 0.5).float()

    # pred_np = pred.detach().cpu().numpy()
    # image_np = pred_np[0, 0, :, :]
    # print(f"Image Np : {image_np}")
    # plt.imshow(image_np)
    # plt.show()

    # # now compute the loss
    # loss = lossfun(yHat, y)

    # # Check is Loss is getting computed
    # print(' ')
    # print('Loss:')
    # print(loss)

    ##############################################
    # Checking the Outputs of the Above Functions
    ##############################################
    # folder = "saved_images"
    # net, lossfun, optimizer = MakeUnet(True)

    # check_accuracy(valid_loader, net, device)
    # save_predictions_as_images(valid_loader, net, folder, device)

    #######################
    # MAIN SCRIPT
    #######################
    load_model = False
    # Saving Validation Images for Visual Inspection
    folder_validation = "/validation_output/"
    # Epochs
    numepochs = 1
    # Extracting Unet, loss function and optimizer Model.
    net, lossfun, optimizer = MakeUnet(False)
    # Pushing the model to GPU.
    net.to(device)

    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), net)

    # Checking Accuracy of the Validation Set before Training.
    print(f"Accuracy of the Validation Set before Training :\n")
    check_accuracy(valid_loader, net, device)

    scaler = torch.cuda.amp.GradScaler()

    for epochi in range(numepochs):

        print(f"Training Loop {epochi+1}")
        # Train the model.
        train_unet_model(train_loader, net, optimizer, lossfun, scaler, device)
        print(f"Finished Training Loop {epochi+1}")
        # Saving the Model {DONE}
        checkpoint = {"state_dict": net.state_dict(),
                "optimizer":optimizer.state_dict(),
                    }
        save_checkpoint(checkpoint)
        # Check Accuracy of the Validation Set. Done at every epoch to see progress.
        print(f"Checking Accuracy of the Validation Set after ")
        check_accuracy(valid_loader, net, device)

        # Save Validation images in a folder.
        save_predictions_as_images(valid_loader, net, folder_validation, device)

    #######################################
    # Checking the Accuracy for Test Set
    #######################################
    folder_test = "/test_output/"
    check_accuracy(test_loader, net, device)
    save_predictions_as_images(test_loader, net,folder_test,device)

    

if __name__ == "__main__":
    main()


