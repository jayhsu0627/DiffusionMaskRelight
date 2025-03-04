#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --time=1-23:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2

# NCCL configuration
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ens3f0  # or the relevant network interface name
export HOST_GPU_NUM=2
export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1
export LOCAL_RANK=${SLURM_LOCALID}  # Set correct local rank

# Debugging
echo "Using LOCAL_RANK=$LOCAL_RANK on GPU $CUDA_VISIBLE_DEVICES"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT= 12340 # Randomize the port to avoid conflicts
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export NUMEXPR_MAX_THREADS=16
echo "MASTER_ADDR="$MASTER_ADDR

## run

srun --ntasks=2 accelerate launch  --multi_gpu train_svd_relight_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid"  \
--mixed_precision='fp16' \
--enable_xformers_memory_efficient_attention \
--video_folder="/fs/gamma-projects/svd_relight/preprocessed_sync" \
--report_to="wandb" \
--learning_rate=3e-5 \
--lr_scheduler="cosine_with_restarts" \
--per_gpu_batch_size=4 \
--gradient_accumulation_steps=4 \
--mixed_precision="fp16" \
--num_train_epochs=8000 \
--output_dir="/fs/gamma-projects/svd_relight/output_mul2" \
--num_workers=1 \
--num_n_frames=16 \
--num_frames=16 \
--checkpointing_steps=1000 \
--width=128 \
--height=128 
# --resume_from_checkpoint='/fs/gamma-projects/svd_relight/output_mul2/checkpoint-7500'