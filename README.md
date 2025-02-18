# DiffusionMaskRelight

# Installation

### 1. Clone this repo
Give a star before you clone this repo please. [![Star on GitHub](https://img.shields.io/github/stars/jonsn0w/hyde.svg?style=social)](https://github.com/jayhsu0627/DiffusionMaskRelight/stargazers)

`git clone git@github.com:jayhsu0627/DiffusionMaskRelight.git`

### 2. Install conda env

```
conda env create -f environment.yml
```

## Train

### Pre-process datasets

We have two datasets loader './utils/dataset.py' and './utils/virtual_dataset.py'.

- MultiIlluminationDataset Class: Directory like this

```
data
└── test
│   ├── everett_dining1
│   ├── everett_dining2
│   ├── everett_kitchen2
│   ├── everett_kitchen4
│   └── ...
└── train
    ├── 14n_copyroom1
    ├── 14n_copyroom6
    ├── 14n_copyroom8
        ├── all_alb.png
        ├── all_depth.png
        ├── all_normal.png
        ├── dir_0_mip2.jpg
        ├── dir_0_mip2_alb.png
        ├── dir_0_mip2_scb.png
        ├── dir_0_mip2_shd.png
        ├── ...
        ├── dir_24_mip2.jpg
        ├── dir_24_mip2_alb.png
        ├── dir_24_mip2_scb.png
        ├── dir_24_mip2_shd.png
        ├── materials_mip2.png
        ├── meta.json
        ├── probes
        │   ├── dir_0_chrome256.jpg
        └── ...
    └── ...
```

- SyncDataset Class: Directory like this

```
test001
├── RGB0000.png
├── ...
├── albedo0000.png
├── ...
├── ao0000.png
├── ...
├── depth0000.png
├── ...
├── mask0000.png
├── ...
├── normal0000.png
├── ...
├── relight0000.png
```
Check dataloader, save loaded images.

```
CUDA_VISIBLE_DEVICES=1 python utils/dataset.py
CUDA_VISIBLE_DEVICES=1 python utils/virtual_dataset.py
```

- Preprocess: We estimated `depth`, `normal` by [Marigold](https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage) by using `preprocess_light_vector_est_MIT.py` (not provided). We generated scribbles (mask), albedo, shading by [Intrinsic Image Decomposition](https://github.com/compphoto/Intrinsic) by using `preprocess_shading_MIT.py` (not provided).

<!-- 
```
python dataloader.py -p data/ -s lego
```

| Frame 0                           | Frame 1                           |
|-----------------------------------|-----------------------------------|
| ![Image 1](./img/data/r_00000.png)    | ![Image 2](./img/data/r_00001.png)    | -->


### Training

- This repo contains three training code: (1) `train_svd.py` for fine-tuning stable video diffusion. (2)`train_svd_relight.py` for fine-tuning MIT datasets. (3)`train_svd_relight_syn.py` for fine-tuning synthetic datasets.

train MIT datasets
```python
CUDA_VISIBLE_DEVICES=1 python train_svd_relight.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/" \
--report_to="wandb" \
--learning_rate=3e-5 \
--lr_scheduler="cosine_with_restarts" \
--per_gpu_batch_size=1 \
--gradient_accumulation_steps=16 \
--mixed_precision="fp16" \
--num_train_epochs=100 \
--output_dir="/sdb5/output_2"
```
or train synthetic data
```python
CUDA_VISIBLE_DEVICES=1 python train_svd_relight_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16' \
--video_folder="/sdb5/test001/" \
--report_to="wandb" \
--learning_rate=3e-5 \
--lr_scheduler="cosine_with_restarts" \
--per_gpu_batch_size=1 \
--gradient_accumulation_steps=16 \
--mixed_precision="fp16" \
--num_train_epochs=100 \
--output_dir="/sdb5/output_3"

```

## Validation

validation view
```python
CUDA_VISIBLE_DEVICES=0 python main.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output/checkpoint-500/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/" \
--output_dir="/sdb5/output"
```

```python
CUDA_VISIBLE_DEVICES=0 python main_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output_3/checkpoint-700/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/test001/" \
--output_dir="/sdb5/output/sync"
```

## Known Issues and TODOs

- [ ] TBD


# Acknowledgements

- **Stability:** for stable video diffusion.
- **Diffusers Team:** For the svd implementation.
- **Pixeli99:** For providing a practical svd training script: [SVD_Xtend](https://github.com/pixeli99/SVD_Xtend)
- **Stable Video Diffusion Temporal Controlnet** For providing the foundation SVD temporal ControlNet code base  [Code](https://github.com/CiaraStrawberry/svd-temporal-controlnet/)
