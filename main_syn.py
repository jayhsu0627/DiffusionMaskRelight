#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
from torch.utils.data import Subset, DataLoader

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
# from diffusers import StableVideoDiffusionPipeline
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler

# Load the added input_ch unet
from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from utils.dataset import MultiIlluminationDataset
from utils.virtual_dataset import SyncDataset

from torch.utils.data import Dataset, random_split
from accelerate.utils import DistributedDataParallelKwargs

import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential
from torchvision import transforms
import kornia
import imageio.v3 as iio

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

#i should make a utility function file
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        # image = image.resize(target_size)
        image = image
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

def create_image_grid(images, rows, cols, target_size=(512, 512)):
    target_size = images[0].size
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder,i, step_num):
    # Flatten batch_output, which is a list of lists of PIL Images
    flattened_batch_output = [img for sublist in batch_output for img in sublist]

    # # Combine frames into a list without converting (since they are already PIL Images)
    # combined_frames = validation_images + validation_control_images + flattened_batch_output

    # Calculate rows and columns for the grid
    num_images = len(validation_images)
    cols = num_images  # adjust number of columns as needed
    rows = 1

    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # filename = f"combined_frames_{step_num}_{i}.png"
    # Create and save the grid image
    grid_org = create_image_grid(validation_images, rows, cols)
    grid_pre = create_image_grid(flattened_batch_output, rows, cols)
    grid_con = create_image_grid(validation_control_images, rows, cols)

    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    
    # Now define the full path for the file
    # timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_org = f"org_{step_num}_{i}.png"
    filename_pre = f"pre_{step_num}_{i}.png"
    filename_con = f"con_{step_num}_{i}.png"

    output_org_loc = os.path.join(output_folder, filename_org)
    output_pre_loc = os.path.join(output_folder, filename_pre)
    output_con_loc = os.path.join(output_folder, filename_con)

    # print(output_pre_loc)
    if grid_org is not None and step_num==1:
        grid_org.save(output_org_loc)

        grid_con.save(output_con_loc)
        # print("grid_org", grid_pre)
        # print("grid_pre", grid_pre)
        # print(output_org_loc)
        # print(output_pre_loc)
    else:
        print("Failed to create image grid")

    if grid_pre is not None:
        # print(output_pre_loc)
        grid_pre.save(output_pre_loc)
    else:
        print("Failed to create image grid")

def convert_colors(image):
    """
    Convert blue and red lines in an image to green, while preserving the white background.

    Parameters:
    image (PIL.Image.Image): The input image object.

    Returns:
    PIL.Image.Image: The modified image with blue and red lines converted to green.
    """
    # Convert image to RGBA if not already
    image = image.convert("RGBA")
    
    # Convert image to a NumPy array
    img_array = np.array(image)
    
    # Create a mask for blue and red pixels
    blue_mask = (img_array[..., 0] == 0) & (img_array[..., 1] == 0) & (img_array[..., 2] == 255)  # Blue pixels
    red_mask = (img_array[..., 0] == 255) & (img_array[..., 1] == 0) & (img_array[..., 2] == 0)   # Red pixels
    
    # # Create a mask for white background pixels
    # white_mask = (img_array[..., :3] == [255, 255, 255]).all(axis=-1)
    
    # Apply color changes
    img_array[blue_mask | red_mask] = [0, 255, 0, 255]  # Convert blue and red to green
    # img_array[white_mask] = [255, 255, 255]        # Preserve white background
    
    # Convert modified array back to a PIL Image
    return Image.fromarray(img_array, "RGBA")

def resize_and_pad_image(image, target_size=(1024, 512), background_color=(255, 255, 255)):

    # # Convert to PIL Image if input is a tensor
    # if isinstance(image, torch.Tensor):
    #     image = kornia.utils.tensor_to_image(image)
    
    # Resize the image while preserving aspect ratio
    width, height = image.size
    target_width, target_height = target_size
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    image = image.resize((new_width, new_height), resample=PIL.Image.BICUBIC)
    
    # Convert the resized image to have a transparent background
    image = image.convert("RGBA")
    
    # Create a new image with the target size and white background
    result = Image.new('RGBA', target_size, background_color)
    
    # Paste the resized image into the center of the new image
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    result.paste(image, (x, y), image)
    
    return result.convert('RGB')

def sort_frames(frame_name):
    # Extract the filename without extension
    # IMG: 0000--020124.png
    # Mask: 0000--020124--0000.png

    frame_name = frame_name.split('.')[0]

    # Split by '--' to separate idx, fr, and maskid (if exists)
    parts = frame_name.split('--')

    # Convert each part to int for sorting
    idx = int(parts[0])  # {idx}
    fr = int(parts[1])   # {fr}

    if len(parts) == 3:  # Case with {maskid}
        maskid = int(parts[2])  # {maskid}
        return (idx, fr, maskid)
    else:  # Case without {maskid}
        return (idx, fr)

def combine_masks(mask_folder):

    combined_masks_list = []
    unique_combinations = []
    # mask_files = os.listdir(mask_folder)
    mask_files = sorted(os.listdir(mask_folder), key=sort_frames)

    # Group masks by idx and fr
    for mask_file in mask_files:
        idx, fr, _ = sort_frames(mask_file)
        key = (idx, fr)
        # print(key)
        # print(os.path.exists(os.path.join(mask_folder, mask_file)))
        mask = iio.imread(os.path.join(mask_folder, mask_file))
        # print(mask.min(), mask.max())
        
        # /sdb2/datasets/WebVid/Don/mask/sakura_quest_s01e007_0130_008763/0130--008718--0000.png

        binary_mask = np.where(mask > 0, 1, 0)  # Convert to binary (0 and 1)
        
        # Check if the combination already exists, and update the mask
        if key in unique_combinations:
            idx_in_list = unique_combinations.index(key)
            combined_masks_list[idx_in_list] = np.logical_or(combined_masks_list[idx_in_list], binary_mask).astype(np.uint8)
        else:
            # Add a new entry for a new combination
            unique_combinations.append(key)
            combined_masks_list.append(binary_mask)
    
    # Stack all the combined masks into a NumPy array (in the order of unique_combinations)
    combined_masks_array = np.stack(combined_masks_list, axis=0)

    # Return the combined masks array and unique combinations
    return combined_masks_array

def load_images_from_folder(folder, mask_folder, is_condition=False):
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}  # Add or remove extensions as needed

    # Function to extract frame number from the filename
    def frame_number(filename):
        parts = filename.split('_')
        if len(parts) > 1 and parts[0] == 'frame':
            try:
                return int(parts[1].split('.')[0])  # Extracting the number part
            except ValueError:
                return float('inf')  # In case of non-integer part, place this file at the end
        return float('inf')  # Non-frame files are placed at the end

    # Sorting files based on frame number
    sorted_files = sorted(os.listdir(folder))

    # Load images in sorted order
    for i,filename in enumerate(sorted_files):

        img = Image.open(os.path.join(folder, filename))

        # Check if the directory exists
        if os.path.isdir(mask_folder):
            mask = combine_masks(mask_folder)[i]
            # Expand mask to 3D to match the shape of image_array (1080, 1920, 3)
            mask_3d = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
            # Convert image to a NumPy array
            image_array = np.array(img)
            multiplied_image_array = (image_array * mask_3d).astype(np.uint8) 

            multiplied_image_array = multiplied_image_array + ((1-mask_3d) * 255).astype(np.uint8) 

            img = Image.fromarray(multiplied_image_array)

        if is_condition:
            img = convert_colors(img)
        w, h = img.size  # PIL uses (width, height) order
        img = resize_and_pad_image(img)

        images.append(img)

    return images



# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def make_train_dataset(args):
    # dataset = MultiIlluminationDataset(args.video_folder,
    #                                     frame_size=25, sample_n_frames=25)
    dataset = SyncDataset(args.video_folder,
                                        frame_size=25, sample_n_frames=25)

    return dataset

def make_test_dataset(args):
    dataset = MultiIlluminationDataset("/sdb5/data/test/",
                                        frame_size=25, sample_n_frames=25)
    return dataset


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    with torch.no_grad():
        latents = vae.encode(t).latent_dist.sample()
        
    del t
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    torch.cuda.empty_cache()  # 🚀 Free GPU memory
    return latents

# def latent_to_tensor(latents, vae, frames=2):

#     video_length = latents.shape[1]
#     print(vae.config.scaling_factor)

#     latents = latents / vae.config.scaling_factor
#     latents = rearrange(latents, "b f c h w -> (b f) c h w")
    
#     t = vae.decode(latents, num_frames = frames).sample
    
#     # dec = self.decode(z, num_frames=num_frames).sample

#     t = rearrange(t, "(b f) c h w -> b f c h w", f=video_length)
    
#     # latents = latents * vae.config.scaling_factor

#     return t

# def latent_to_tensor(latents, vae, frames=2):
#     video_length = latents.shape[1]
#     latents = latents / vae.config.scaling_factor
    
#     # Process in smaller batches
#     batch_size = 4  # Adjust based on your GPU memory
#     decoded_frames = []
    
#     for i in range(0, video_length, batch_size):
#         batch_latents = latents[:, i:i+batch_size]
#         batch_latents = rearrange(batch_latents, "b f c h w -> (b f) c h w")
        
#         with torch.no_grad():  # Optional: if you don't need gradients
#             decoded = vae.decode(batch_latents, num_frames=frames).sample
        
#         decoded = rearrange(decoded, "(b f) c h w -> b f c h w", f=min(batch_size, video_length-i))
#         decoded_frames.append(decoded)
    
#     # Concatenate all batches
#     t = torch.cat(decoded_frames, dim=1)
#     return t

# def latent_to_tensor(latents, vae, frames=2):
#     video_length = latents.shape[1]
#     latents = latents / vae.config.scaling_factor
    
#     # Process in smaller batches
#     batch_size = 4  # Adjust based on your GPU memory
#     decoded_frames = []
    
#     for i in range(0, video_length, batch_size):
#         batch_latents = latents[:, i:i+batch_size]
#         current_batch_size = batch_latents.shape[1]  # Actual number of frames in this batch
#         batch_latents = rearrange(batch_latents, "b f c h w -> (b f) c h w")
        
#         with torch.no_grad():  # Optional: if you don't need gradients
#             # Make sure num_frames matches the current batch
#             decoded = vae.decode(batch_latents, num_frames=current_batch_size).sample
        
#         decoded = rearrange(decoded, "(b f) c h w -> b f c h w", f=current_batch_size)
#         decoded_frames.append(decoded)
    
#     # Concatenate all batches
#     t = torch.cat(decoded_frames, dim=1)
#     return t

def latent_to_tensor(latents, vae, frames=2):

    video_length = latents.shape[1]
    # print(vae.config.scaling_factor)

    latents = latents / vae.config.scaling_factor
    latents = rearrange(latents, "b f c h w -> (b f) c h w")
    
    with torch.no_grad():
        t = vae.decode(latents, num_frames = frames).sample
    
    del latents  # 🚀 Free `latents` after decoding
    t = rearrange(t, "(b f) c h w -> b f c h w", f=video_length)
    torch.cuda.empty_cache()  # 🚀 Free GPU memory
    return t


def save_latents_as_images(latents, output_dir):
    """
    Save latents as a series of images.

    Args:
        latents (torch.Tensor): The latent tensor to save (b, f, c, h, w).
        output_dir (str): The directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert the tensor to a NumPy array and ensure it's on CPU
    latents_np = latents.cpu().detach().numpy()  # Shape: (b, f, c, h, w)

    # Iterate through the batch and frames
    for b in range(latents_np.shape[0]):
        for f in range(latents_np.shape[1]):
            latent_frame = latents_np[b, f]
            print("latent_frame shape:", latent_frame.shape)

            # Collapse the channels if needed (e.g., average for grayscale)
            if latent_frame.shape[0] >= 3:  # RGB latent
                latent_frame = latent_frame.transpose(1, 2, 0)  # Convert to HWC
            else:  # Assume single-channel latent
                latent_frame = latent_frame[0]

            # Normalize the latent frame to [0, 255] for saving
            latent_frame = (latent_frame - latent_frame.min()) / (latent_frame.max() - latent_frame.min()) * 255
            latent_frame = latent_frame.astype(np.uint8)

            # Save the frame ase an image
            image_path = os.path.join(output_dir, f"latent_b{b}_f{f}.png")
            latent_frame = cv2.cvtColor(latent_frame, cv2.COLOR_RGB2BGR)

            cv2.imwrite(image_path, latent_frame)
            print(f"Saved latent frame as image: {image_path}")

def save_generated_images(tensor, output_dir, pre):
    """
    Save generated images from a tensor.

    Args:
        tensor (torch.Tensor): The tensor containing generated images (b, f, c, h, w).
        output_dir (str): Directory to save the images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensor to NumPy array and move to CPU
    images_np = tensor.cpu().detach().numpy()  # Shape: (b, f, c, h, w)

    # Iterate over batches and frames
    for b in range(images_np.shape[0]):
        for f in range(images_np.shape[1]):
            image = images_np[b, f]  # Shape: (c, h, w)

            # Convert channels from CHW to HWC
            if image.shape[0] >= 3:  # RGB image
                image = image.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:  # Grayscale image
                image = image[0]  # Use the first channel

            # Normalize to [0, 255] for saving
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)

            # Save the image
            image_path = os.path.join(output_dir, f"image_b{b}_f{f}_"+ str(pre) +".png")
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image)
            print(f"Saved image: {image_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=20,
        help=(
            "Run fine-tuning test every X epochs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=12345, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--pretrain_vae",
        type=str,
        default=None,
        help="use weight for unet block",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help=(
            "path to the dataset csv"
        ),
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help=(
            "path to the video folder"
        ),
    )
    parser.add_argument(
        "--condition_folder",
        type=str,
        default=None,
        help=(
            "path to the depth folder"
        ),
    )
    parser.add_argument(
        "--motion_folder",
        type=str,
        default=None,
        help=(
            "path to the motion folder"
        ),
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=None,
        help=(
            "path to the mask folder"
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image_num",
        type=int,
        default=None,
        help=(
            "num of validation sets"
        ),
    )

    parser.add_argument(
        "--validation_image_folder",
        type=str,
        default=None,
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_control_folder",
        type=str,
        default=None,
        help=(
            "the validation control image"
        ),
    )
    parser.add_argument(
        "--validation_msk_folder",
        type=str,
        default=None,
        help=(
            "the validation mask image"
        ),
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=15,
        help=(
            "frames per video"
        ),
    )
    parser.add_argument(
        "--test_motion_value",
        type=int,
        default=9999,
        help=(
            "motion_bucket"
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def pil_image_to_numpy(image):
    img = image
    image = img.resize((img.size[0]//2, img.size[1]//2))

    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))

    return images.float() / 255

def is_all_white(image):
    return np.all(image == 255)

def blend_tensors(rgb_tensor, mask_tensor, blend_ratio=0.5):
    """
    Blend RGB tensor with mask tensor to create grayscale output.
    
    Parameters:
    rgb_tensor (torch.Tensor): RGB tensor of shape (B, F, 3, H, W)
    mask_tensor (torch.Tensor): Mask tensor of shape (B, F, 1, H, W)
    blend_ratio (float): Ratio for blending (0.0 to 1.0), default is 0.5
    
    Returns:
    torch.Tensor: Blended grayscale tensor of shape (B, F, 1, H, W)
    """
    # Convert RGB to grayscale using ITU-R BT.601 conversion
    # Coefficients: R: 0.299, G: 0.587, B: 0.114
    rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 1, 3, 1, 1).to(rgb_tensor.device)
    rgb_gray = torch.sum(rgb_tensor * rgb_weights, dim=2, keepdim=True)
    
    # # Ensure tensors are in float format
    # rgb_gray = rgb_gray.float()
    # mask_tensor = mask_tensor.float()
    
    # Blend the tensors
    blended = blend_ratio * rgb_gray + (1 - blend_ratio) * mask_tensor
    
    return blended

def main():

    args = parse_args()
    accelerator = Accelerator()
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")


    # # Create an instance of the model
    # model = UNetSpatioTemporalConditionModel()

    # # Print the input channels
    # print("Input channels:", model.conv_in.in_channels)

    # # print(vars(UNetSpatioTemporalConditionModel()) )

    # # to be trained
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True
    )
    
        # ignore_mismatched_sizes=True

    # vae_enc = AutoencoderKLTemporalDecoder.from_pretrained(
    #     args.pretrained_model_name_or_path  if args.pretrain_vae is None else args.pretrain_vae,
    #     subfolder="vae")

    print("VAE precision:", next(vae.parameters()).dtype)
    print("UNet precision:", next(unet.parameters()).dtype)
    # print("Trainable VAE precision:", next(vae_enc.parameters()).dtype)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print("weight_dtype:", weight_dtype)

    # Move image_encoder and vae to gpu and cast to weight_dtype
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    # vae_enc.to(accelerator.device, dtype=weight_dtype)

    #controlnet.to(accelerator.device, dtype=weight_dtype)
    # Create EMA for the unet.
    # if args.use_ema:
    #     ema_controlnet = EMAModel(unet.parameters(
    #     ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError(
    #             "xformers is not available. Make sure it is installed correctly")

    # test_dataset = make_test_dataset(args)
    test_dataset = make_train_dataset(args)

    # Use regular DataLoader for test set, without shuffling
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))
            
    preprocessed_dir = "/sdb5/DiffusionMaskRelight/vae_val"
    # output_type = 'g_buffer'
    # output_type = 'rgb'
    output_type = 'rgb'

    def natural_sort_key(s):
        import re
        # Extract the number from filenames like 'dir_10_mip2.jpg'
        match = re.match(r'dir_(\d+)_mip2\.jpg', s)
        if match:
            return int(match.group(1))  # Return the number for sorting
        return float('inf')  # Put non-matching files at the end

    def process_depth_image(file_path):
        # Load the depth image
        depth_image = Image.open(file_path)
        depth_array = np.array(depth_image)
        
        # Normalize the depth for visualization
        depth_normalized = depth_array / 65535.0
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        
        # Save the normalized depth image as 8-bit
        depth_8bit_image = Image.fromarray(depth_8bit)
        # depth_8bit_image.save("/fs/nexus-scratch/sjxu/svd-temporal-controlnet/depth_8bit.png")
        print('depth shape,', depth_8bit_image.size)
        return depth_8bit_image

    def load_images_with_depth_processing(preprocessed_dir, image_files):
        numpy_images = []

        for img in image_files:
            img_path = os.path.join(preprocessed_dir, img)
            
            if 'depth' in img:
                depth_8bit_image = process_depth_image(img_path)
                print('depth_8bit_image', depth_8bit_image.size)
                
                numpy_image = pil_image_to_numpy(depth_8bit_image)
                print('depth numpy', numpy_image.shape)
            else:
                numpy_image = pil_image_to_numpy(Image.open(img_path))
                print('normal numpy', numpy_image.shape)
            
            numpy_images.append(numpy_image)
        
        return np.array(numpy_images)

    # # Sort and limit the number of image and condition files to 14
    # if output_type == 'rgb':
    #     image_files = sorted(os.listdir(preprocessed_dir), key=natural_sort_key)[:24]
    # elif output_type == 'g_biffer':
    #     image_files = [file for file in sorted(os.listdir(preprocessed_dir)) if file in ('all_normal.png', 'all_depth.png')]
    # elif output_type == 'concat':
    #     image_files = [file for file in sorted(os.listdir(preprocessed_dir)) if file in ('all_normal.png', 'all_depth.png')]

    # print('here:', image_files)

    # # Load image frames
    


    # # numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
    # numpy_images = load_images_with_depth_processing(preprocessed_dir, image_files)
    # print('numpy_images:', numpy_images.shape)


    # pixel_values = numpy_to_pt(numpy_images).to(weight_dtype).to(
    #     accelerator.device, non_blocking=True
    # )
    # pixel_values = pixel_values.unsqueeze(0)
    # print("pixel_values", pixel_values.shape)
    #     # else:
    #     #     pixel_values = batch["pixel_values"].to(weight_dtype).to(
    #     #         accelerator.device, non_blocking=True
    #     #     )

    # latents = tensor_to_vae_latent(pixel_values, vae).to(weight_dtype).to(
    #     accelerator.device, non_blocking=True
    # )
    # # encoded
    # save_generated_images(pixel_values, "/sdb5/DiffusionMaskRelight/outputs/" + str(output_type), "org")

    # # latent space
    # save_latents_as_images(latents, "/sdb5/DiffusionMaskRelight/outputs/" + str(output_type))

    # # decoded

    # recon_pix_values = latent_to_tensor(latents, vae)
    # save_generated_images(recon_pix_values, "/sdb5/DiffusionMaskRelight/outputs/" + str(output_type), "rec")


    # The models need unwrapping because for compatibility in distributed training mode.
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        image_encoder=image_encoder,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)

    for batch in test_dataloader:
    
        # pixel_values, depths, normals, albedos, scribbles = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                     batch["depth_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                     batch["normal_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                     batch["alb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                     batch["scb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True)

        # pixel_values, depths, normals, albedos, scribbles, shading_gt = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                                 batch["depth_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                                 batch["normal_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                                 batch["alb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                                 batch["scb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
        #                                                                 batch["shd_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True)

        # relight images, depth, normal, albedo, mask, original images
        pixel_values, depths, normals, albedos, scribbles, rgbs       = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
                                                                        batch["depth_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
                                                                        batch["normal_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
                                                                        batch["alb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
                                                                        batch["scb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True), \
                                                                        batch["rgb_pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True)

        input_image = albedos[:, 0:1, :, :, :]

        print(input_image[0].shape)
        # input_image = (input_image+1)/2

        # # 2nd, repeat 1-ch to 3-ch for depth and scribbles

        # # Add encoded images to trainable vae

        # mask_blended = blend_tensors(albedos, scribbles, blend_ratio=0.6).to(weight_dtype).to(accelerator.device, non_blocking=True)
        # mask_blended = mask_blended.repeat(1, 1, 3, 1, 1)  # Expand along dim=2 to 3 channels                
        # mask_blended = rearrange(mask_blended, "b f c h w -> (b f) c h w")
        # mask_blended = F.interpolate(mask_blended, scale_factor=0.5, mode="bilinear", align_corners=False)
        # mask_blended = rearrange(mask_blended, "(b f) c h w -> b f c h w", f=albedos.shape[1])

        #     # encode (0.5*mask + 0.5*albedo_gray) **3 to improve the 1st ch
        # mask_latents = tensor_to_vae_latent(mask_blended, vae_enc)
        #     # decode (0.5*mask + 0.5*albedo_gray) **3 to get the 1st ch
        # recon_shd = latent_to_tensor(mask_latents, vae_enc, frames = mask_latents.shape[1])
        
        # vae_enc.to("cpu")  # Move VAE to CPU after encoding

        # recon_shd = rearrange(recon_shd, "b f c h w -> (b f) c h w")
        # recon_shd = F.interpolate(recon_shd, scale_factor=2.0, mode="bilinear", align_corners=False)
        # recon_shd = rearrange(recon_shd, "(b f) c h w -> b f c h w", f=albedos.shape[1])

        # shading = recon_shd[:, :, 0:1, :, :]

        # # 🚀 Free memory after use
        # del mask_latents, mask_blended

        print("input", input_image.shape)
        video_frames = pipeline(
            input_image[0],
            # g_buffer= [depths, normals, albedos, shading],
            g_buffer= [rgbs, depths, normals, albedos, scribbles],
            height=256,
            width=256,
            num_frames= 25,
            decode_chunk_size=8,
            motion_bucket_id=127,
            fps=7,
            noise_aug_strength=0.02,
            generator=generator,
        ).frames[0]

        # # 🚀 Free memory after use
        # del shading

        val_save_dir = os.path.join(
            args.output_dir, "validation_images")

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)

        out_file = os.path.join(
            val_save_dir,
            f"val_img_800_syn_l.mp4",
        )
        out_file_gt = os.path.join(
            val_save_dir,
            f"val_gt_800_syn_l.mp4",
        )

        for i in range(25):
            img = video_frames[i]
            video_frames[i] = np.array(img)
        export_to_gif(video_frames, out_file, 8)
        
        # # recon_shd = recon_shd.squeeze(0).detach().cpu().numpy()

        # video_frames = [0] *25
        # for i in range(25):
        #     img_np = recon_shd[0][i].detach().cpu().numpy()  # Convert to NumPy array
        #     # img_np = np.transpose(img_np, (1, 2, 0))
        #     img_np = img_np[0]
        #     if img_np.ndim == 2:  
        #         img_np = np.stack([img_np] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)
        #     # print(img_np.shape)
        #     img_np = (img_np * 255).astype(np.uint8)
        #     video_frames[i] = img_np
        # export_to_gif(video_frames, out_file, 8)

        # video_frames = [0] *25

        # for i in range(25):
        #     img_np = shading_gt[0][i].detach().cpu().numpy()  # Convert to NumPy array
        #     # img_np = np.transpose(img_np, (1, 2, 0))
        #     img_np = img_np[0]
        #     if img_np.ndim == 2:  
        #         img_np = np.stack([img_np] * 3, axis=-1)  # Convert (H, W) → (H, W, 3)
        #     # print(img_np.shape)

        #     # img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
        #     img_np = (img_np * 255).astype(np.uint8)
        #     video_frames[i] = img_np

        for i in range(25):
            img_np = pixel_values[0][i].detach().cpu().numpy()  # Convert to NumPy array
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np * 255).astype(np.uint8)
            video_frames[i] = img_np
        export_to_gif(video_frames, out_file_gt, 8)

        break
        # concatenate image


if __name__ == "__main__":
    main()
