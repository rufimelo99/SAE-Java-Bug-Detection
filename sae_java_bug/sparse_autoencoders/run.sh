#!/bin/bash

#SBATCH --job-name=saes
#SBATCH --mem=30G

#SBATCH --gres=shard:0
#SBATCH --time=300:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=all
#SBATCH --mail-user=rufimelo99@gmail.com
#SBATCH --output=/home/u021521/depois-ve-se/lurm-%x-%j.out
#SBATCH --error=/home/u021521/depois-ve-se/slurm-%x-%j.err

# Prepare Environment
source activate /home/u021521/anaconda3/envs/myenv/
echo "Submitting job"


BASE_DIR=/home/u021521/depois-ve-se/sae_java_bug/sparse_autoencoders

for i in {0..11}; do
  python3 sae_exploration.py \
      --csv_path artifacts/humaneval.csv \
      --layer $i \
      --model gpt2-small \
      --sae_id blocks.$i.hook_resid_pre \
      --release gpt2-small-res-jb \
      --cache_component hook_resid_pre.hook_sae_acts_post \
      --output_dir $BASE_DIR/gpt2_humaneval/layer$i
  python3 vulnerability_detection_features.py --dir-path gpt2_humaneval/layer$i/
done

# for i in {1..24};
# do
#   python sae_exploration.py \
#     --csv_path $BASE_DIR/artifacts/defects4j.csv \
#     --layer $i \
#     --model google/gemma-2-2b \
#     --release gemma-scope-2b-pt-res-canonical \
#     --sae_id layer_${i}/width_16k/canonical \
#     --cache_component hook_resid_post.hook_sae_acts_post \
#     --output_dir $BASE_DIR/gemma2_defects4j/layer$i

# for i in {1..24};
# do
#   python vulnerability_detection_features.py --dir-path gemma2_defects4j/layer$i/
# done