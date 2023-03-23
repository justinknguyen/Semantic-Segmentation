#Import required libraries
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16_bn, VGG16_BN_Weights
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
    
def create_video(image_dir):
    # Define the output video file name
    output_file = "input_video.mp4"

    # Get the list of image file names
    image_files = sorted([file for file in os.listdir(image_dir) if file.endswith(".png")])
    # Keep only the first half of the image files
    image_files = image_files[:len(image_files) // 2]

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width, _ = first_image.shape

    # Set the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 10, (width, height))

    # Loop through the image files and add them as frames to the video
    for file in image_files:
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()

def process_video(input_file, output_file, net, device):
    cap = cv2.VideoCapture(input_file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to match the required dimensions
        resized_frame = cv2.resize(frame, (448, 448))

        # Normalize the frame
        image = get_transforms()(resized_frame.copy() / 255.)

        # Convert the frame to a torch tensor
        x_tensor = image[None].float().to(device)

        # Predict the mask
        with torch.no_grad():
            _mask = net(x_tensor)
            _, _mask = torch.max(_mask, dim=1)

        # Apply the colormap to the predicted mask
        num_classes = 30
        _mask = _mask.squeeze().cpu().numpy()
        mask_3d = np.repeat(_mask[:, :, np.newaxis], 3, axis=2)
        color_mask = cv2.applyColorMap((mask_3d * 255 / (num_classes - 1)).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize the color mask back to the original frame size
        color_mask = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Combine the original frame and the color mask
        output_frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

        # Write the frame
        out.write(output_frame)

    # Release the resources
    cap.release()
    out.release()
    
def create_ground_truth_video(image_folder, mask_folder, output_file, fps=10):
    image_files = sorted(os.listdir(image_folder))
    image_files = image_files[:len(image_files) // 2]
    mask_files = sorted(os.listdir(mask_folder))
    mask_files = image_files[:len(mask_files) // 2]

    # Get dimensions from the first image
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image_file, mask_file in zip(image_files, mask_files):
        # Read the ground truth mask
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)

        # Combine the mask layers
        num_classes = 30
        combined_mask = np.zeros_like(mask)
        for i in range(num_classes):
            combined_mask += ((mask == i) * i * 255 / num_classes).astype(np.uint8)

        # Apply a colormap to the combined mask
        color_mask = cv2.applyColorMap(combined_mask.astype(np.uint8), cv2.COLORMAP_JET)

        # Read the corresponding image and blend it with the color mask
        image = cv2.imread(os.path.join(image_folder, image_file))
        blended_frame = cv2.addWeighted(image, 1, color_mask, 0.5, 0)

        # Write the blended frame to the video
        video_writer.write(blended_frame)

    # Release the resources
    video_writer.release()
    
def display_frames_side_by_side(original_video, output_video, ground_truth_video, frame_indices):
    original_cap = cv2.VideoCapture(original_video)
    output_cap = cv2.VideoCapture(output_video)
    gt_cap = cv2.VideoCapture(ground_truth_video)

    total_frames = int(original_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in frame_indices:
        if idx < total_frames:
            original_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            output_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            gt_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            _, original_frame = original_cap.read()
            _, output_frame = output_cap.read()
            _, gt_frame = gt_cap.read()

            # Convert frames to RGB format
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)

            # Plot the frames side-by-side
            fig, axs = plt.subplots(1, 3, figsize=(30, 30))
            axs[0].imshow(original_frame)
            axs[0].set_title('Original Frame')
            axs[1].imshow(gt_frame)
            axs[1].set_title('Ground Truth Frame')
            axs[2].imshow(output_frame)
            axs[2].set_title('Predicted Frame')

            # Remove axis ticks
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            plt.show()

    original_cap.release()
    output_cap.release()
    gt_cap.release()