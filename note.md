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
--mixed_precision='bf16' \
--video_folder="/sdb5/data/train/"
```

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
--num_train_epochs=200 \
--output_dir="/sdb5/output_3" \
--resume_from_checkpoint="/sdb5/output_3/checkpoint-800"


validation view
```python
CUDA_VISIBLE_DEVICES=0 python main.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output/shd_control/checkpoint-10/unet/" \
--pretrain_vae="/sdb5/output/shd_control/checkpoint-2000/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/" \
--output_dir="/sdb5/output"

```
CUDA_VISIBLE_DEVICES=0 python main.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output_2/checkpoint-1500/unet/" \
--pretrain_vae="/sdb5/output_2/checkpoint-1500/vae/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/" \
--output_dir="/sdb5/output"

CUDA_VISIBLE_DEVICES=0 python main.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output/checkpoint-500/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/data/train/" \
--output_dir="/sdb5/output"

CUDA_VISIBLE_DEVICES=0 python main_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output_3/checkpoint-700/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/test001/" \
--output_dir="/sdb5/output/sync"

CUDA_VISIBLE_DEVICES=0 python main_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--pretrain_unet="/sdb5/output_4/checkpoint-800/unet/" \
--mixed_precision='fp16' \
--video_folder="/sdb5/test001/" \
--output_dir="/sdb5/output/sync"

# depth image is 16bit

CUDA_VISIBLE_DEVICES=1 python utils/dataset.py
CUDA_VISIBLE_DEVICES=1 python preprocess_shading_MIT.py
`preprocess_light_vector_est_MIT` to process depth and normal, `preprocess_shading_MIT` to process the albedo, shading, and scribbles.
