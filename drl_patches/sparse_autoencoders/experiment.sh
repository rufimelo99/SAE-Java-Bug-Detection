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


echo "Getting experiment config. Gettig train indexes"
python getting_experiment_config.py --csv_path artifacts/gbug-java.csv --output_path artifacts/gbug-java_train_indexes.json
python getting_experiment_config.py --csv_path artifacts/defects4j.csv --output_path artifacts/defects4j_train_indexes.json
python getting_experiment_config.py --csv_path artifacts/humaneval.csv --output_path artifacts/humaneval_train_indexes.json

run_pipeline() {
    dataset=$1
    csv_path=$2
    train_indexes=$3

    echo "Running pipeline for $dataset from $csv_path with train indexes $train_indexes"

    for i in {0..11}; do
        output_dir=$BASE_DIR/gpt2_${dataset}/layer$i
        python3 sae_exploration.py \
            --csv_path $csv_path \
            --layer $i \
            --model gpt2-small \
            --sae_id blocks.$i.hook_resid_pre \
            --release gpt2-small-res-jb \
            --cache_component hook_resid_pre.hook_sae_acts_post \
            --output_dir $output_dir

        python3 vulnerability_detection_features.py \
            --dir-path gpt2_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes \
            --save-model
    done
}

# Run for each dataset
echo "Running pipeline for gbug-java"
run_pipeline "gbug-java" "artifacts/gbug-java.csv" "artifacts/gbug-java_train_indexes.json"
echo "Running pipeline for defects4j"
run_pipeline "defects4j" "artifacts/defects4j.csv" "artifacts/defects4j_train_indexes.json"
echo "Running pipeline for humaneval"
run_pipeline "humaneval" "artifacts/humaneval.csv" "artifacts/humaneval_train_indexes.json"


run_bert_pipeline() {
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
run_bert_pipeline "microsoft/graphcodebert-base"
run_bert_pipeline "answerdotai/ModernBERT-base"
run_bert_pipeline "answerdotai/ModernBERT-large"




python baselines/gather_hidden_states.py --csv_path artifacts/gbug-java.csv --output_dir gemma2_hidden_states_gbug-java --model_name google/gemma-2-2b
python baselines/gather_hidden_states.py --csv_path artifacts/defects4j.csv --output_dir gemma2_hidden_states_defects --model_name google/gemma-2-2b
python baselines/gather_hidden_states.py --csv_path artifacts/humaneval.csv --output_dir gemma2_hidden_states_humaneval --model_name google/gemma-2-2b


# Baselines
# Getting the vectorizer
echo "Getting the vectorizer"
python get_vectorizer.py --csvs artifacts/gbug-java.csv artifacts/humaneval.csv artifacts/defects4j.csv --output_dir artifacts/
python classical_data_mining.py --csv_path artifacts/defects4j.csv --output_dir ole --train-indexes_path artifacts/defects4j_train_indexes.json  --vectorizer_path artifacts/vectorizer.pkl