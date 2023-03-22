#Import required libraries
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_snippets import stems, read
from random import randint

def get_transforms():
  # Dataset is large, don't necessarily need to apply random data augmentation
  # Mean and std dev based on average of the training data
  return transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.285, 0.322, 0.282],
                                 [0.176, 0.181, 0.178]
                                 )
                             ])
  
class SegmentationDataset(Dataset):
    def __init__(self, split):
        self.items = stems(f'dataset/image_{split}')
        self.split = split


    def __len__(self):
        return len(self.items)


    def __getitem__(self, ix):
        image = read(f'dataset/image_{self.split}/{self.items[ix]}.png', 1)
        image = cv2.resize(image, (448,448))

        mask = read(f'dataset/mask_{self.split}/{self.items[ix]}.png')
        mask = cv2.resize(mask, (448,448))

        return image, mask
    
    def getOne(self): return self[randint(0, len(self)-1)]
    
    def collate_fn(self, batch):
        images, masks = list(zip(*batch))
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Apply transformations
        images = torch.cat([get_transforms()(im.copy()/255.)[None] for im in images]).float().to(device)

        ground_truth_masks = torch.cat([torch.Tensor(mask)[None] for mask in masks]).long().to(device)


        return images, ground_truth_masks
    
from torchvision.models import vgg16_bn, VGG16_BN_Weights

# Define the convolutional block
def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# Define the up-convolutional block
def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, out_channels=34):
        super().__init__()

        # Use VGG16_BN as a starting point for the encoder half of the UNet
        self.encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])


        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        # Manually define the decoder portion
        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
    
    # Build up the forward pass from the components defined above
    def forward(self, x):
        
        
        # Left half of UNet - encoding
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)


        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        # Right half of UNet - encoding
        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)


        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)


        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)


        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)


        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)


        x = self.conv11(x)


        return x