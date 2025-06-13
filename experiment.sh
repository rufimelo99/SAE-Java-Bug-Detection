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

        python3 getting_sorted_layer_features.py \
            --dir-path gpt2_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes

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


run_gemma2b_pipeline() {
    dataset=$1
    csv_path=$2
    train_indexes=$3

    echo "Running pipeline for $dataset from $csv_path with train indexes $train_indexes"

    for i in {0..24}; do
        output_dir=$BASE_DIR/gemma2_${dataset}/layer$i
        
        python sae_exploration.py \
            --csv_path $csv_path \
            --layer $i \
            --model google/gemma-2-2b \
            --release gemma-scope-2b-pt-res-canonical \
            --sae_id layer_$i/width_16k/canonical \
            --cache_component hook_resid_post.hook_sae_acts_post \
            --output_dir $BASE_DIR/gbug-java/layer$i

        python3 getting_sorted_layer_features.py \
            --dir-path gemma2_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes

        python3 vulnerability_detection_features.py \
            --dir-path gemma2_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes \
            --save-model
    done
}

# Run for each dataset
echo "Running pipeline for gbug-java"
run_gemma2b_pipeline "gbug-java" "artifacts/gbug-java.csv" "artifacts/gbug-java_train_indexes.json"
echo "Running pipeline for defects4j"
run_gemma2b_pipeline "defects4j" "artifacts/defects4j.csv" "artifacts/defects4j_train_indexes.json"
echo "Running pipeline for humaneval"
run_gemma2b_pipeline "humaneval" "artifacts/humaneval.csv" "artifacts/humaneval_train_indexes.json"


# for i in {1..33};
# do
#   python3 sae_exploration.py \
#     --csv_path $BASE_DIR/artifacts/defects4j.csv \
#     --layer $i \
#     --model meta-llama/Llama-3.1-8B \
#     --release llama_scope_lxr_32x \
#     --sae_id l${i}r_32x \
#     --cache_component hook_resid_post.hook_sae_acts_post \
#     --output_dir $BASE_DIR/llama_defects4j/layer$i
# done




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

run_hidden_states_pipeline() {
    model=$1
    csv_path=$2
    output_dir=$3

    echo "Running pipeline for $model, $csv_path, $output_dir"

    python baselines/gather_hidden_states.py \
        --csv_path $csv_path \
        --output_dir $output_dir \
        --model_name $model

}

# Run for each dataset
run_hidden_states_pipeline openai-community/gpt2 artifacts/gbug-java.csv gpt2_hidden_states_gbug-java
run_hidden_states_pipeline openai-community/gpt2 artifacts/defects4j.csv gpt2_hidden_states_defects
run_hidden_states_pipeline openai-community/gpt2 artifacts/humaneval.csv gpt2_hidden_states_humaneval

run_hidden_states_pipeline google/gemma-2-2b artifacts/gbug-java.csv gemma2_hidden_states_gbug-java
run_hidden_states_pipeline google/gemma-2-2b artifacts/defects4j.csv gemma2_hidden_states_defects
run_hidden_states_pipeline google/gemma-2-2b artifacts/humaneval.csv gemma2_hidden_states_humaneval



run_hidden_states_prediction_pipeline() {
    local model_prefix=$1     # e.g., "gpt2" or "gemma2"
    local layers=$2           # number of layers, e.g., 12 for GPT-2, 25 for Gemma2
    local train_indexes=$3    # path to train index file
    local dataset=$4          # dataset name

    echo "Running $model_prefix hidden states pipeline for $dataset, with $train_indexes"

    for ((i = 0; i < layers; i++)); do
        output_dir=$BASE_DIR/${model_prefix}_hidden_states_${dataset}/layer$i

        python3 getting_sorted_layer_features.py \
            --dir-path ${model_prefix}_hidden_states_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes

        python3 vulnerability_detection_features.py \
            --dir-path ${model_prefix}_hidden_states_${dataset}/layer$i/ \
            --train-indexes_path $train_indexes \
            --save-model
    done
}

# Define datasets and their training index files
declare -A datasets
datasets=(
    ["gbug-java"]="artifacts/gbug-java_train_indexes.json"
    ["defects4j"]="artifacts/defects4j_train_indexes.json"
    ["humaneval"]="artifacts/humaneval_train_indexes.json"
)

# Run for GPT-2 (12 layers)
for dataset in "${!datasets[@]}"; do
    run_hidden_states_prediction_pipeline "gpt2" 12 "${datasets[$dataset]}" "$dataset"
done

# Run for Gemma2B (25 layers)
for dataset in "${!datasets[@]}"; do
    run_hidden_states_prediction_pipeline "gemma2" 25 "${datasets[$dataset]}" "$dataset"
done



# Baselines


# Run for each dataset
run_bert_pipeline "microsoft/graphcodebert-base"
run_bert_pipeline "answerdotai/ModernBERT-base"
run_bert_pipeline "answerdotai/ModernBERT-large"



# Getting the vectorizer
echo "Getting the vectorizer"
python get_vectorizer.py --csvs artifacts/gbug-java.csv artifacts/humaneval.csv artifacts/defects4j.csv --output_dir artifacts/
python classical_data_mining.py --csv_path artifacts/defects4j.csv --output_dir ole --train-indexes_path artifacts/defects4j_train_indexes.json  --vectorizer_path artifacts/vectorizer.pkl