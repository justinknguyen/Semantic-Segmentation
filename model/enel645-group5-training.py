#Import required libraries
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_snippets import stems, read
from torch.optim.lr_scheduler import ExponentialLR

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print("The device being used is:", device)

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
    
    def collate_fn(self, batch):
        images, masks = list(zip(*batch))
        
        # Apply transformations
        images = torch.cat([get_transforms()(im.copy()/255.)[None] for im in images]).float().to(device)

        ground_truth_masks = torch.cat([torch.Tensor(mask)[None] for mask in masks]).long().to(device)


        return images, ground_truth_masks
    
train_ds = SegmentationDataset('train')
val_ds = SegmentationDataset('val')

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=train_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=8, collate_fn=val_ds.collate_fn)

# Use VGG16_BN as a starting point for the encoder half of the UNet
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

        # Use VGG16_BN for the encoder portion
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

net = UNet().to(device)

criterion = nn.CrossEntropyLoss() # Loss function
optimizer = torch.optim.AdamW(net.parameters(), lr = 0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9) # decrease learning rate over time

nepochs = 100
PATH = './best_model.pth' # Path to save the best model
last_model_update = 0

best_loss = 1e+20
for epoch in range(nepochs):  # loop over the dataset multiple times
    # Training Loop
    train_loss = 0.0
    net.train()
    for i, data in enumerate(train_dl):
        images, ground_truth_masks = data
        optimizer.zero_grad()
        
        # forward + backward + optimize
        _masks = net(images)
        
        loss = criterion(_masks, ground_truth_masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
    print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')
    scheduler.step()
    
    val_loss = 0
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_dl):
            images, ground_truth_masks = data
            _masks = net(images)
            loss = criterion(_masks, ground_truth_masks)
            
            val_loss += loss.item()
            
        print(f'val loss: {val_loss / i:.3f}')
        
        last_model_update += 1
        
        # Save best model
        if val_loss < best_loss:
            print("Saving model")
            torch.save(net.state_dict(), PATH)
            best_loss = val_loss
            last_model_update = 0
            
        # Early stopping if model doesn't improve in 10 epochs
        if last_model_update >= 10:
            print("Stopping early. No model improvement in 10 epochs.")
            break
            
print('Finished Training')