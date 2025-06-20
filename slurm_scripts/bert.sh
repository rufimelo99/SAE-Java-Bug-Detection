#!/bin/bash

#SBATCH --job-name=bert
#SBATCH --mem=50G

#SBATCH --gres=shard:20
#SBATCH --time=300:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=rufimelo99@gmail.com
#SBATCH --output=/home/u021521/ARSI/logs/lurm-%x-%j.out
#SBATCH --error=/home/u021521/ARSI/logs/slurm-%x-%j.err
# Prepare Environment
source activate /home/u021521/anaconda3/envs/cl/
echo "Submitting job"


BASE_DIR=/home/u021521/depois-ve-se/sae_java_bug/sparse_autoencoders/
cd $BASE_DIR

run_pipeline() {
    model=$1

    echo "Running pipeline for $model"

    python bert.py  \
        --dataset artifacts/gbug-java.csv  \
        --training_indices artifacts/gbug-java_train_indexes.json \
        --model $model

    python bert.py  \
        --dataset artifacts/defects4j.csv  \
        --training_indices artifacts/defects4j_train_indexes.json \
        --model $model

    python bert.py  \
        --dataset artifacts/humaneval.csv  \
        --training_indices artifacts/humaneval_train_indexes.json \
        --model $model
}

# Run for each dataset
run_pipeline "microsoft/graphcodebert-base"
run_pipeline "answerdotai/ModernBERT-base"
run_pipeline "answerdotai/ModernBERT-large"
