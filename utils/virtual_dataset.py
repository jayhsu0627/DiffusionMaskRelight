
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


import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

def pil_image_to_numpy(image):
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

def save_array_as_image(array, filename):
    # Ensure the array has the correct data type (uint8 for images)
    if array.dtype != np.uint8:
        array = array.numpy().astype(np.uint8)
    # array = (array + 1)
    array = array.transpose(1, 2, 0)

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
    # Convert the array to an image using PIL
    img = Image.fromarray(array, mode="L")
    
    # Save the image to the specified filename
    img.save(filename)

class SyncDataset(Dataset):
    def __init__(
            self,
            root_dir, frame_size=25, sample_n_frames=14, mixed_precision=""
        ):
        # zero_rank_print(f"loading annotations from {csv_path} ...")
        
        self.root_dir = root_dir
        self.frame_size = frame_size
        self.weight_dtype= torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Find all scene directories
        self.scene_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

        self.dataset = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)    
        # self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        
        # sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        # print("sample size",sample_size)
        
        self.transforms_0 = ImageSequential(
            K.SmallestMaxSize(256, p=1.0),
            K.CenterCrop((256, 256)),
            same_on_batch=True  # This enables getting the transformation matrices
        )

    def __len__(self):
        return len(self.dataset)

    def sort_frames(self, frame_name):
        # Extract the numeric part from the filename
        # dir_0_mip2.jpg
        frame_name = frame_name.split('.')[0]
        parts = frame_name.split('_')
        # print('parts', parts)
        if len(parts) == 3:
            return int(parts[1])
        else:
            return 9999

    def sort_frames_scb(self, frame_name):
        # Extract the numeric part from the filename
        # dir_0_mip2_scb.png
        frame_name = frame_name.split('.')[0]
        parts = frame_name.split('_')
        suffix_real= parts[-1]

        # print('parts', parts)
        if len(parts) > 3:
            if suffix_real == 'scb':
                return int(parts[1])
            else:
                return 999
        else:
            return 9999

    def sort_frames_shd(self, frame_name):
        # Extract the numeric part from the filename
        # dir_0_mip2_scb.png
        # dir_0_mip2_shd.png
        # dir_0_mip2_alb.png

        frame_name = frame_name.split('.')[0]
        parts = frame_name.split('_')
        suffix_real= parts[-1]

        # print('parts', parts)
        if len(parts) > 3:
            if suffix_real == 'shd':
                return int(parts[1])
            else:
                return 999
        else:
            return 9999

    def sort_frames_sync(self, frame_name):
        # Extract the filename without extension
        base_name = frame_name.split('.')[0]
        
        # Check if the filename has the format with underscores
        if '_' not in base_name:
            # Find where the digits start
            i = 0
            while i < len(base_name) and not base_name[i].isdigit():
                i += 1
            
            # Extract the numeric part
            if i < len(base_name):
                digits = ''
                while i < len(base_name) and base_name[i].isdigit():
                    digits += base_name[i]
                    i += 1
                
                if digits:
                    return int(digits)

        # Default return if no valid numeric part is found
        return 9999

    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split('_')[1].split('.')[0])
    
        while True:
            video_dict = self.dataset[idx]
            # videoid = os.path.basename(video_dict)
            # preprocessed_dir = os.path.join(self.root_dir, videoid)
            preprocessed_dir = os.path.join(self.root_dir)

            # relit_folder  = glob.glob(os.path.join(preprocessed_dir, "relight*.png"))  # depth should have one image
            depth_folder  = glob.glob(os.path.join(preprocessed_dir, "depth*.png"))  # depth should have one image
            normal_folder = glob.glob(os.path.join(preprocessed_dir, "normal*.png"))  # depth should have one image
            # alb_folder    = glob.glob(os.path.join(preprocessed_dir, "albedo*.png"))  # depth should have one image

            # print(preprocessed_dir)
            # print(depth_folder)
            # print(normal_folder)
            # print(alb_folder)

            if not os.path.exists(depth_folder[0]) or not os.path.exists(normal_folder[-1]):
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Sort and limit the number of image and depth files to 14
            input_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'RGB'in f][:self.sample_n_frames]
            image_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'relight'in f][:self.sample_n_frames]
            depth_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'depth'in f][:self.sample_n_frames]
            normal_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'normal'in f][:self.sample_n_frames]
            alb_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'albedo'in f][:self.sample_n_frames]

            # Choose scribbles
            scb_files = [f for f in sorted(os.listdir(preprocessed_dir), key=self.sort_frames_sync) if 'mask'in f][:self.sample_n_frames]
            # shd_files = sorted(os.listdir(preprocessed_dir), key=self.sort_frames_shd)[:self.sample_n_frames]
            
            # print('relight', image_files, len(image_files))
            # print('dep', depth_files, len(depth_files))
            # print('nrm', normal_files, len(normal_files))
            # print('alb', alb_files, len(alb_files))
            # print('scb', scb_files, len(scb_files))
            # print('RGB', input_files, len(input_files))

            # Check if there are enough frames for both image and depth
            if len(image_files) < self.sample_n_frames or len(depth_files) < self.sample_n_frames:
                idx = random.randint(0, len(self.dataset) - 1)
                continue
    
            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)
            # pixel_values = pixel_values*2 -1

            # Load depth frames
            numpy_depth_images = np.array([((np.array(Image.open(os.path.join(preprocessed_dir, df))))).astype(np.uint8) for df in depth_files])
            # numpy_depth_images = np.stack([numpy_depth_images] * 3, axis=-1)
            # print(numpy_depth_images.shape)

            depth_pixel_values = numpy_to_pt(numpy_depth_images)
            # depth_pixel_values = depth_pixel_values*2 -1

            # Load normal frames
            numpy_nrm_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, nm))) for nm in normal_files])
            normal_pixel_values = numpy_to_pt(numpy_nrm_images)
            # normal_pixel_values = normal_pixel_values*2 -1

            # Load alb frames
            numpy_alb_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, alb))) for alb in alb_files])
            alb_pixel_values = numpy_to_pt(numpy_alb_images)
            # alb_pixel_values = alb_pixel_values*2 -1

            # Load scb frames
            numpy_scb_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in scb_files])
            scb_pixel_values = numpy_to_pt(numpy_scb_images)
            # scb_pixel_values = scb_pixel_values*2 -1

            # Load shd frames
            numpy_RGB_images = np.array([pil_image_to_numpy(Image.open(os.path.join(preprocessed_dir, img))) for img in input_files])
            RGB_pixel_values = numpy_to_pt(numpy_RGB_images)

            pixel_values = self.transforms_0(pixel_values)
            depth_pixel_values = self.transforms_0(depth_pixel_values)
            normal_pixel_values = self.transforms_0(normal_pixel_values)
            alb_pixel_values = self.transforms_0(alb_pixel_values)
            scb_pixel_values = self.transforms_0(scb_pixel_values)
            RGB_pixel_values = self.transforms_0(RGB_pixel_values)

            return pixel_values, depth_pixel_values[:, 0:1, :, :], normal_pixel_values, alb_pixel_values, scb_pixel_values[:, 0:1, :, :], RGB_pixel_values

     
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, RGB_pixel_values = self.get_batch(idx)
        
        sample = dict(pixel_values=pixel_values.to(dtype=self.weight_dtype),
                    depth_pixel_values=depth_pixel_values.to(dtype=self.weight_dtype),
                    normal_pixel_values=normal_pixel_values.to(dtype=self.weight_dtype),
                    alb_pixel_values=alb_pixel_values.to(dtype=self.weight_dtype),
                    scb_pixel_values=scb_pixel_values.to(dtype=self.weight_dtype),
                    rgb_pixel_values=RGB_pixel_values.to(dtype=self.weight_dtype),
                    )
        return sample

if __name__ == "__main__":
    # from utils.util import save_videos_grid

    dataset = SyncDataset(root_dir="/sdb5/test001",
                                        frame_size=31, sample_n_frames=31)

    for i in range(1):
        idx = np.random.randint(len(dataset))
        pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, rgb_pixel_values = dataset.get_batch(idx)

    print(pixel_values.shape)
    print(depth_pixel_values.shape)
    print(normal_pixel_values.shape)
    print(alb_pixel_values.shape)
    print(scb_pixel_values.shape)
    print(rgb_pixel_values.shape)

    save_array_as_image(pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/rgb_01.png")
    save_array_as_image(pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/rgb_10.png")

    save_array_as_image_depth(depth_pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/dep_01.png")
    save_array_as_image_depth(depth_pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/dep_10.png")
    
    save_array_as_image(normal_pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/nrm_01.png") 
    save_array_as_image(normal_pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/nrm_10.png")

    save_array_as_image(alb_pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/alb_01.png")
    save_array_as_image(alb_pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/alb_10.png")

    save_array_as_image_depth(scb_pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/scb_01.png")
    save_array_as_image_depth(scb_pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/scb_10.png")

    save_array_as_image(rgb_pixel_values[0]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/shd_01.png")
    save_array_as_image(rgb_pixel_values[-1]*255, "/sdb5/DiffusionMaskRelight/outputs/sync/shd_10.png")

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1,)
    # # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)

    # for idx, batch in enumerate(dataloader):
    #     print(batch["pixel_values"].shape)
    #     print(batch["depth_pixel_values"].shape)
    #     print(batch["normal_pixel_values"].shape)
    #     print(batch["alb_pixel_values"].shape)
    #     print(batch["scb_pixel_values"].shape)

    #     break

    # for i in range(2):
    #     idx = np.random.randint(len(dataset))
    #     train_image, train_depth, train_normal = dataset.get_batch(idx)

    # print('length:', len(dataset))
    # print(train_image.shape, train_depth.shape, train_normal.shape)
