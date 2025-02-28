#!/bin/bash

#SBATCH --job-name=saes
#SBATCH --mem=20G
#SBATCH --output=/home/u021521/depois-ve-se/logs/job_out_%A_%a.out

#SBATCH --gres=shard:4
#SBATCH --time=4:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=rufimelo99@gmail.com
#SBATCH --output=/home/u021521/depois-ve-se/lurm-%x-%j.out
#SBATCH --error=/home/u021521/depois-ve-se/slurm-%x-%j.err

# Prepare Environment
source activate /home/u021521/anaconda3/envs/myenv/
echo "Submitting job"


BASE_DIR=/home/u021521/depois-ve-se/drl_patches/sparse_autoencoders

for i in {1..1};
do
  python sae_exploration.py \
    --csv_path $BASE_DIR/artifacts/defects4j.csv \
    --layer $i \
    --model google/gemma-2-2b \
    --release gemma-scope-2b-pt-res-canonical \
    --sae_id layer_$i/width_16k/canonical \
    --cache_component hook_resid_post.hook_sae_acts_post \
    --output_dir $BASE_DIR/gbug-java/layer$i
done