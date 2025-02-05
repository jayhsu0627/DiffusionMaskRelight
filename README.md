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