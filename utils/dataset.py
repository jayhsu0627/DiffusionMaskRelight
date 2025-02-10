# import os, io, csv, math, random, json
# import glob
# import numpy as np
# from einops import rearrange
# import re
# from collections import defaultdict

# import torch
# from decord import VideoReader
# import cv2

# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# from torch.utils.data.dataset import Dataset
# import sys
# # Add the project root to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.util import zero_rank_print
# #from torchvision.io import read_image
# from PIL import Image, ImageOps
# import imageio.v3 as iio
# import torch.nn.functional as F
# import kornia.augmentation as K
# from kornia.augmentation.container import ImageSequential
# # from relighting.light_directions import get_light_dir_encoding, BACKWARD_DIR_IDS

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True



# def pil_image_to_numpy(image):
#     """Convert a PIL image to a NumPy array."""
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     width, height = image.size
#     aspect_ratio = width / height
    
#     new_width = 512
#     new_height = int(new_width /aspect_ratio)

#     # Resize the image
#     new_size = (new_width, new_height)  # Specify the desired width and height
#     image = image.resize(new_size)
    
#     return np.array(image)


# def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
#     """Convert a NumPy image to a PyTorch tensor."""
#     if images.ndim == 3:
#         images = images[..., None]
#     # print("numpy_to_pt", images.shape)
#     images = torch.from_numpy(images.transpose(0, 3, 1, 2))
#     return images.float() / 255

# INPUT_IDS = torch.tensor([
#         49406, 49407,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#         0,     0,     0,     0,     0,     0,     0
#         ] # this is a tokenized version of the empty string
#         )

# class MIL(Dataset):
#     def __init__(
#             self, video_folder,condition_folder,motion_folder,
#             sample_size=256, sample_n_frames=14,
#         ):

#         self.json = [
#             json.loads(line) for line in open(f"relighting/training_edit.json", "r").read().splitlines()
#         ]

#         self.dataset = self.json
#         # self.length = len(self.dataset)
#         self.length = len(self.json)

#         print(f"data scale: {self.length}")
#         random.shuffle(self.dataset)    
#         self.video_folder    = video_folder
#         self.sample_n_frames = sample_n_frames
#         self.condition_folder = condition_folder
#         self.motion_values_folder=motion_folder
#         print("length",len(self.dataset))
#         sample_width = sample_size
#         sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
#         print("sample size",sample_size)

#         self.groups = {
#             "A": [0, 1, 4, 5, 6, 7],
#             "B": [8, 9, 10, 11, 12, 13],
#             "C": [14, 15, 16, 17, 18, 23],
#         }

#         # Crop operation
#         self.transforms_0 = ImageSequential(
#             K.CenterCrop((256, 512)),
#             same_on_batch=True  # This enables getting the transformation matrices
#         )

#     def __len__(self):
#         return self.length

#     def sort_frames(operationself, frame_name):
#         # Extract the numeric part from the filename
#         # dir_0_mip2.jpg
#         frame_name = frame_name.split('.')[0]
#         parts = frame_name.split('_')
#         # print('parts', parts)
#         if len(parts) > 2:
#             return int(parts[1])
#         else:
#             return 9999
#             # return int(parts[1][5:])

#     def group_filenames_by_set(self, filenames, video_id):
#         # Prepare a dictionary to hold filtered filenames by group
#         grouped_files = defaultdict(list)
        
#         # Pattern to extract frame number from filename (e.g., 14n_office8_dir_0.jpg -> 0)
#         pattern = re.compile(f"dir_(\d+)")
#         # Iterate over filenames and match them to groups
#         for filename in filenames:
#             match = pattern.search(filename)
#             if match:
#                 frame_num = int(match.group(1))  # Extract the frame number
                
#                 # Check which group this frame number belongs to and add it to the group
#                 for group_name, frame_range in self.groups.items():
#                     if frame_num in frame_range:
#                         grouped_files[group_name].append(filename)
#                         break  # Stop once we find the correct group
        
#         return grouped_files

#     def get_batch(self, idx):        
#         # for blender random

#         while True:
#             video_dict = self.dataset[idx]
#             videoid = video_dict['video_id']
#             # print(videoid)

#             video_id = self.json[idx]["video_id"].replace(".jpg", ".png")
#             target_set = self.json[idx]["target_image"].replace(".jpg", ".png")
#             cond_set = self.json[idx]["conditioning_image"].replace(".jpg", ".png")
#             depth_set = 'all_depth.png'
#             normal_set = 'all_normal.png'
#             image_path = self.video_folder + "/" + video_id

#             if not os.path.exists(image_path):
#                 print('continue path valid')
#                 idx = random.randint(0, len(self.dataset) - 1)
#                 continue    

#             # print(image_path)
#             # filenames = sorted([f for f in os.listdir(image_path) if f.endswith(".jpg")], key=self.sort_frames)[:self.sample_n_frames]
#             filenames = sorted([f for f in os.listdir(image_path) if f.endswith(".jpg")], key=self.sort_frames)

#             # video_id = "14n_office8"
#             # print(filenames)

#             grouped_files = self.group_filenames_by_set(filenames, video_id)
#             image_files = []
#             cond_files = []
#             # print("grouped_files",grouped_files)
#             # Display the grouped files
#             for group_name, files in grouped_files.items():
#                 # print(group_name, target_set)                
#                 if group_name == target_set:
#                     # print(f"target_set {group_name}: {files}")
#                     image_files = files
#                 if group_name == cond_set:
#                     # print(f"cond_set {group_name}: {files}")
#                     cond_files = files
#             # if image_files == []: continue
#             # if cond_files == []: continue

#             # target_dir = [get_light_dir_encoding(int(img_dir.split("_")[1])) for img_dir in image_files]
#             # target_dir = torch.from_numpy(np.array(target_dir))
#             # print(target_dir.shape)

#             # Check if there are enough frames for both image and depth
#             if len(image_files) < self.sample_n_frames or len(cond_files) < self.sample_n_frames:
#                 print(len(image_files),len(image_files) < self.sample_n_frames, len(cond_files) )
#                 print('continue length')
#                 idx = random.randint(0, len(self.dataset) - 1)
#                 continue

#             # Load image frames
#             numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(image_path, img)).convert("RGB")) for img in image_files])
#             pixel_values = numpy_to_pt(numpy_images)
#             height, width = numpy_images[0].shape[:2]

#             # Load control frames
#             numpy_control_images = np.array([pil_image_to_numpy(Image.open(os.path.join(image_path, cond)).convert("RGB")) for cond in cond_files])
#             cond_pixel_values = numpy_to_pt(numpy_control_images)

#             # Load depth frames
#             # Read in 16 bit depth png
#             # print(os.path.join(image_path, depth_set))
#             numpy_depth_images = np.array([((np.array(Image.open(os.path.join(image_path, depth_set)))/65535.0)* 255).astype(np.uint8) for cond in cond_files])
#             numpy_depth_images = np.stack([numpy_depth_images] * 3, axis=-1)
#             depth_pixel_values = numpy_to_pt(numpy_depth_images)
#             depth_pixel_values = F.interpolate(depth_pixel_values, size=(height, width), mode='bilinear', align_corners=False)

#             # Load normal frames
#             numpy_normal_images = np.array([pil_image_to_numpy(Image.open(os.path.join(image_path, normal_set)).convert("RGB")) for cond in cond_files])
#             normal_pixel_values = numpy_to_pt(numpy_normal_images)
#             normal_pixel_values = F.interpolate(normal_pixel_values, size=(height, width), mode='bilinear', align_corners=False)

#             motion_values = [5]
#             motion_values = torch.from_numpy(np.array(motion_values))

#             frame_size = pixel_values.shape[0]
#             # print(len(image_files),len(cond_files))
#             # print(depth_pixel_values.shape, normal_pixel_values.shape)
#             combined = self.transforms_0(torch.cat([pixel_values, cond_pixel_values, depth_pixel_values, normal_pixel_values], dim=0))

#             pixel_values, cond_pixel_values, depth_pixel_values, normal_pixel_values = combined[:frame_size], combined[frame_size: frame_size*2], combined[frame_size*2: frame_size*3], combined[frame_size*3:]
#             return pixel_values, cond_pixel_values, motion_values, depth_pixel_values[:, 0:1, :, :], normal_pixel_values

#     def __getitem__(self, idx):
        
#         pixel_values, cond_pixel_values, motion_values, depth_pixel_values, normal_pixel_values = self.get_batch(idx)
#         # pixel_values = self.pixel_transforms(pixel_values)

#         sample = dict(  text="",
#                         pixel_values=pixel_values,
#                         condition_pixel_values=cond_pixel_values,
#                         depth_pixel_values=depth_pixel_values,
#                         motion_values=motion_values,
#                         input_ids = INPUT_IDS,
#                         )
#         return sample



# if __name__ == "__main__":
#     from utils.util import save_videos_grid

#     dataset = MIL(
#         video_folder="/sdb5/data/train",
#         condition_folder = "/fs/nexus-scratch/sjxu/WebVid/blender_random/shd",
#         motion_folder = "/fs/nexus-scratch/sjxu/WebVid/blender_random/motion",
#         sample_size=512,
#         sample_n_frames=6
#         )

#     # idx = np.random.randint(len(dataset))
#     # train_image, train_cond, _, train_depth, train_normal, _ = dataset.get_batch(idx)
    
#     for i in range(2):
#         idx = np.random.randint(len(dataset))
#         train_image, train_cond, _, train_depth, train_normal, train_dir = dataset.get_batch(idx)


#     print('length:', len(dataset))
#     print(train_image.shape, train_cond.shape, train_depth.shape, train_dir.shape)

#     save_array_as_image(train_image[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_0.png')
#     save_array_as_image(train_cond[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_0.png')
#     save_array_as_image_depth(train_depth[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_depth_image_0.png')
#     save_array_as_image(train_normal[0]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_normal_image_0.png')
    
#     save_array_as_image(train_image[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_1.png')
#     save_array_as_image(train_cond[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_1.png')
#     save_array_as_image_depth(train_depth[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_depth_image_1.png')
#     save_array_as_image(train_normal[1]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_normal_image_1.png')

#     save_array_as_image(train_image[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_image_2.png')
#     save_array_as_image(train_cond[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_cond_image_2.png')
#     save_array_as_image_depth(train_depth[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_depth_image_2.png')
#     save_array_as_image(train_normal[2]*255, '/fs/nexus-scratch/sjxu/svd-temporal-controlnet/output_normal_image_2.png')

import os
import glob
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from PIL import Image
from collections import defaultdict
import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential

def sort_frames(frame_name):
    # Extract the numeric part from the filename
    # dir_0_mip2.jpg
    frame_name = frame_name.split('.')[0]
    parts = frame_name.split('_')
    # print('parts', parts)
    if len(parts) > 2:
        return int(parts[1])
    else:
        return 9999

def save_array_as_image(array, filename):
    
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    array = array.transpose(1, 2, 0)
    print("rgb",array.shape)

    # Convert the array to an image using PIL
    img = Image.fromarray(array)
    
    # Save the image to the specified filename
    img.save(filename)

def save_array_as_image_depth(array, filename):
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    array = array.transpose(1, 2, 0)[:,:,0]
    print(array.shape)
    # Convert the array to an image using PIL
    img = Image.fromarray(array, mode="L")
    
    # Save the image to the specified filename
    img.save(filename)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    # print("numpy_to_pt", images.shape)
    # images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    images = torch.from_numpy(images)

    return images.float() / 255

# ==========================
# 1. Dataset Class
# ==========================
class MultiIlluminationDataset(Dataset):
    def __init__(self, root_dir, frame_size=10):
        """
        Args:
            root_dir (str): Root directory containing scene folders.
            transform (callable, optional): Optional transformations.
        """
        self.root_dir = root_dir
        self.frame_size = frame_size

        # Find all scene directories
        self.scene_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.data = []  # Store (scene_name, illumination_image, depth_image) pairs
        
        # Crop operation
        self.transforms_0 = ImageSequential(
            K.CenterCrop((256, 256)),
            same_on_batch=True  # This enables getting the transformation matrices
        )

        for scene_dir in self.scene_dirs:
            scene_name = os.path.basename(scene_dir)
            illumination_images = sorted([f for f in os.listdir(scene_dir) if f.find("dir_*.jpg")], key=sort_frames)[:self.frame_size]
            illumination_images = [os.path.join(scene_dir,file) for file in illumination_images]

            depth_image = glob.glob(os.path.join(scene_dir, "*_depth.png"))  # depth should have one image
            norm_image = glob.glob(os.path.join(scene_dir, "*_normal.png"))  # depth should have one image

            if len(illumination_images) > 0 and len(depth_image) == 1:
                # Repeat the depth image to match the number of illuminations
                repeated_depths = [depth_image[0]] * len(illumination_images)
                repeated_normals = [norm_image[0]] * len(illumination_images)

                self.data.extend(zip([scene_name] * len(illumination_images), illumination_images, repeated_depths, repeated_normals))

        print(f"Loaded {len(self.data)} image pairs from {len(self.scene_dirs)} scenes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene, image_path, depth_path, norm_path = self.data[idx]
        
        file_name = os.path.basename(image_path)

        # Load images
        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path)
        depth = ((np.array(depth)/65535.0)* 255).astype(np.uint8)
        depth = numpy_to_pt(depth)
        normal = Image.open(norm_path).convert("RGB")

        # # numpy_depth_images = np.array([((np.array(Image.open(os.path.join(image_path, depth_set)))/65535.0)* 255).astype(np.uint8) for cond in cond_files])

        # Apply transformations
        if self.transforms_0:
            image = self.transforms_0(image)
            depth = self.transforms_0(depth)
            normal = self.transforms_0(normal)

        # return image, depth, normal, scene, file_name  # Return scene name for batching

        sample = dict(  text="",
                        pixel_values=image,
                        depth_pixel_values=depth,
                        normal_pixel_values=normal,
                        scene=scene,
                        file_name=file_name,
                        )
        return sample

# ==========================
# 2. Scene-Based Sampler
# ==========================
class SceneBatchSampler(Sampler):
    def __init__(self, dataset, frame_size):
        """
        Custom sampler to group batches by scene.
        Args:
            dataset (Dataset): Instance of MultiIlluminationDataset.
            frame_size (int): Number of samples per batch.
        """
        self.frame_size = frame_size
        self.scene_dict = defaultdict(list)

        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset  # Get the original dataset
            
        # Group indices by scene
        for idx, (_, _, _, scene) in enumerate(dataset.data):
            self.scene_dict[scene].append(idx)
        
        # print(self.scene_dict)

        # Convert scene groups into a list of batches
        self.batches = []
        for scene, indices in self.scene_dict.items():
            # random.shuffle(indices)  # Shuffle within scene
            for i in range(0, frame_size, frame_size):
                self.batches.append(indices[i:i + frame_size])

        random.shuffle(self.batches)  # Shuffle batches across scenes

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

# # ==========================
# # 4. Initialize Dataset & DataLoader
# # ==========================
# frame_size = 25
# frame_sample = 25

# # Load dataset
# train_dataset = MultiIlluminationDataset(root_dir="/sdb5/data/train/",
#                                         frame_size=frame_size)

# # train_dataset = MultiIlluminationDataset(root_dir="/sdb5/data/train/", frame_size = frame_size)

# # Use custom batch sampler
# sampler = SceneBatchSampler(train_dataset, frame_sample)
# # sampler = RandomSampler(train_dataset)

# # Create DataLoader
# dataloader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)

# # ==========================
# # 5. Test the DataLoader
# # ==========================
# for batch in dataloader:
#     images, depths, normals, scenes, file = batch["pixel_values"], batch["depth_pixel_values"], batch["normal_pixel_values"], batch["scene"], batch["file_name"]
#     # Frame x batch x ch x W x H
#     print(images.shape)
#     print(f"Batch Scene: {scenes[0]} - Images Size: {len(images)} - Depths Size: {len(depths)} - {file}")
#     print(images[0].min(), images[0].max())
#     print(images[0].shape, depths[0].shape)

#     save_array_as_image(images[0][0]*255, "/sdb5/DiffusionMaskRelight/outputs/rgb_01.png")
#     save_array_as_image(images[-1][0]*255, "/sdb5/DiffusionMaskRelight/outputs/rgb_10.png")

#     save_array_as_image(normals[0][0]*255, "/sdb5/DiffusionMaskRelight/outputs/nrm_01.png")

#     save_array_as_image_depth(depths[0][0]*255, "/sdb5/DiffusionMaskRelight/outputs/dep_01.png")
#     save_array_as_image_depth(depths[-1][0]*255, "/sdb5/DiffusionMaskRelight/outputs/dep_10.png")

#     break