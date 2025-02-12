# DiffusionMaskRelight

```python
python train_svd.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16'  \
--enable_xformers_memory_efficient_attention  \
--allow_tf32 \
--scale_lr \
--lr_scheduler='cosine_with_restarts' \
--use_8bit_adam
```

```python
CUDA_VISIBLE_DEVICES=0,1 python main.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/"
```

```python
CUDA_VISIBLE_DEVICES=1 python train_svd_relight.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/"
```

# depth image is 16bit

CUDA_VISIBLE_DEVICES=1 python utils/dataset.py
CUDA_VISIBLE_DEVICES=1 python preprocess_shading_MIT.py
`preprocess_light_vector_est_MIT` to process depth and normal, `preprocess_shading_MIT` to process the albedo, shading, and scribbles.