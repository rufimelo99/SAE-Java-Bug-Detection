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