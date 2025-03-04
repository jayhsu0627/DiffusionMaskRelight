import os
import glob
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import kornia.augmentation as K
from kornia.augmentation.container import ImageSequential

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


# Helper functions for image conversion
def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    print(images.shape)
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    # return images.float() / 255  # Normalize to [0,1]

    return (images.float() / 127.5 -1).to(torch.float16)  # Normalize to [0,1]

class PreprocessAndSaveDataset:
    def __init__(self, root_dir, save_dir, sample_n_frames=14):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.sample_n_frames = sample_n_frames

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

        # List all scene directories
        self.scene_dirs = sorted(glob.glob(os.path.join(root_dir, "*")))

        # Define transformations
        self.transforms = ImageSequential(
            K.SmallestMaxSize(128, p=1.0),
            K.CenterCrop((128, 128)),
            same_on_batch=True
        )

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

    def process_and_save(self):
        """Preprocess and save all dataset files as .pkl"""
        # for idx, scene_path in enumerate(self.scene_dirs):
        for idx, scene_path in enumerate(self.scene_dirs):

            # Extract scene name
            scene_name = os.path.basename(scene_path)
            save_path = os.path.join(self.save_dir, f"{scene_name}_small.pkl")

            # if os.path.exists(save_path):
            #     print(f"Skipping {scene_name}, already processed.")
            #     continue

            print(scene_path)

            # Collect file lists
            image_files = sorted(glob.glob(os.path.join(scene_path, "relight*.png")))[:self.sample_n_frames]
            depth_files = sorted(glob.glob(os.path.join(scene_path, "depth*.png")))[:self.sample_n_frames]
            normal_files = sorted(glob.glob(os.path.join(scene_path, "normal*.png")))[:self.sample_n_frames]
            alb_files = sorted(glob.glob(os.path.join(scene_path, "albedo*.png")))[:self.sample_n_frames]
            scb_files = sorted(glob.glob(os.path.join(scene_path, "mask*.png")))[:self.sample_n_frames]
            rgb_files = sorted(glob.glob(os.path.join(scene_path, "RGB*.png")))[:self.sample_n_frames]

            # print(image_files)
            # print(depth_files)
            # print(normal_files)
            # print(alb_files)
            # print(scb_files)
            # print(rgb_files)

            # break
            
            # Load and convert images
            def load_images(file_list):
                return numpy_to_pt(np.array([pil_image_to_numpy(Image.open(f)) for f in file_list]))

            data_dict = {
                "pixel_values": self.transforms(load_images(image_files)),
                "depth_pixel_values": self.transforms(load_images(depth_files))[:, 0:1, :, :],  # Convert depth to 1-channel
                "normal_pixel_values": self.transforms(load_images(normal_files)),
                "alb_pixel_values": self.transforms(load_images(alb_files)),
                "scb_pixel_values": self.transforms(load_images(scb_files))[:, 0:1, :, :],  # Convert to 1-channel
                "rgb_pixel_values": self.transforms(load_images(rgb_files))
            }

            # Save as .pkl
            with open(save_path, "wb") as f:
                pickle.dump({k: v.to(torch.float16) for k, v in data_dict.items()}, f)
            
            print(f"Processed and saved: {scene_name}")

        print("All files processed and saved.")

class SyncDataset(Dataset):
    def __init__(self, preprocessed_dir, max_frames=None, mixed_precision="fp16"):
        """
        Dataset class to load preprocessed `.pkl` files.
        
        Args:
            preprocessed_dir (str): Path to directory containing `.pkl` files.
            max_frames (int, optional): Maximum number of frames to load. 
                                        If None, loads all frames.
            mixed_precision (str, optional): Precision format (fp16, bf16, or default fp32).
        """
        self.preprocessed_dir = preprocessed_dir
        self.file_list = sorted(glob.glob(os.path.join(preprocessed_dir, "*.pkl")))
        print(self.file_list)
        self.max_frames = max_frames  # New variable to limit frame count
        self.weight_dtype = torch.float32

        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        print(f"Loaded {len(self.file_list)} preprocessed samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with open(file_path, "rb") as f:
            data_dict = pickle.load(f)

        # Clip to max_frames if specified
        if self.max_frames is not None:
            data_dict = {key: value[:self.max_frames] for key, value in data_dict.items()}

        # Convert data to correct dtype
        sample = {k: v.to(dtype=self.weight_dtype) for k, v in data_dict.items()}
        return sample

# Example usage:
if __name__ == "__main__":
    
    preprocessor = PreprocessAndSaveDataset(
        root_dir="/fs/gamma-projects/svd_relight/sync_data",
        save_dir="/fs/gamma-projects/svd_relight/preprocessed_sync",
        sample_n_frames=31
    )
    preprocessor.process_and_save()

    dataset = SyncDataset(preprocessed_dir="/fs/gamma-projects/svd_relight/preprocessed_sync", max_frames=15)

    # Test loading a batch
    sample = dataset[0]
    print(sample["pixel_values"].shape)  # Check dimensions

    pixel_values, depth_pixel_values, normal_pixel_values, alb_pixel_values, scb_pixel_values, rgb_pixel_values = sample["pixel_values"], sample["depth_pixel_values"], sample["normal_pixel_values"], sample["alb_pixel_values"], sample["scb_pixel_values"], sample["rgb_pixel_values"]

    print(pixel_values.shape)
    print(depth_pixel_values.shape)
    print(normal_pixel_values.shape)
    print(alb_pixel_values.shape)
    print(scb_pixel_values.shape)
    print(rgb_pixel_values.shape)
    pixel_values = (pixel_values+1)/2
    depth_pixel_values = (depth_pixel_values+1)/2
    normal_pixel_values = (normal_pixel_values+1)/2
    alb_pixel_values = (alb_pixel_values+1)/2
    scb_pixel_values = (scb_pixel_values+1)/2
    rgb_pixel_values = (rgb_pixel_values+1)/2

    save_array_as_image(pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/rgb_01.png")
    save_array_as_image(pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/rgb_06.png")
    save_array_as_image(pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/rgb_10.png")

    save_array_as_image_depth(depth_pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/dep_01.png")
    save_array_as_image_depth(depth_pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/dep_06.png")
    save_array_as_image_depth(depth_pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/dep_10.png")
    
    save_array_as_image(normal_pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/nrm_01.png") 
    save_array_as_image(normal_pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/nrm_06.png") 
    save_array_as_image(normal_pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/nrm_10.png")

    save_array_as_image(alb_pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/alb_01.png")
    save_array_as_image(alb_pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/alb_06.png")
    save_array_as_image(alb_pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/alb_10.png")

    save_array_as_image_depth(scb_pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/scb_01.png")
    save_array_as_image_depth(scb_pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/scb_06.png")
    save_array_as_image_depth(scb_pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/scb_10.png")

    save_array_as_image(rgb_pixel_values[0]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/shd_01.png")
    save_array_as_image(rgb_pixel_values[5]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/shd_06.png")
    save_array_as_image(rgb_pixel_values[-1]*255, "/fs/nexus-scratch/sjxu/DiffusionMaskRelight/outputs/sync/shd_10.png")

    print('done')