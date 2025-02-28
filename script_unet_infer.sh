#!/bin/bash
#SBATCH --mem=122G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1:00:00
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

# NCCL configuration
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ens3f0  # or the relevant network interface name
export HOST_GPU_NUM=1
export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
# export LOCAL_RANK=${SLURM_LOCALID}  # Set correct local rank

# # Debugging
# echo "Using LOCAL_RANK=$LOCAL_RANK on GPU $CUDA_VISIBLE_DEVICES"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=$((12340 + RANDOM % 1000))  # Randomize the port to avoid conflicts
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export NUMEXPR_MAX_THREADS=16
echo "MASTER_ADDR="$MASTER_ADDR

## run
srun python main_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
--pretrain_unet="/fs/gamma-projects/svd_relight/output/checkpoint-4200/unet/" \
--mixed_precision='fp16' \
--video_folder="/fs/gamma-projects/svd_relight/sync_data/test001/" \
--output_dir="/fs/nexus-scratch/sjxu/Model_out/sync" \
--num_workers=1

python main_syn.py  \
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
--pretrain_unet="/fs/gamma-projects/svd_relight/output_mul/checkpoint-7500/unet/" \
--mixed_precision='fp16' \
--video_folder="/fs/gamma-projects/svd_relight/sync_data/test001/" \
--output_dir="/fs/nexus-scratch/sjxu/Model_out/sync_fast" \
--num_workers=1