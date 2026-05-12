#!/bin/bash
declare -A my_map

# Add key-value pairs
my_map["run_20260227_215424_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=27

my_map["run_20260227_215007_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=23

my_map["run_20260227_214539_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=19

my_map["run_20260227_214103_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=15

my_map["run_20260227_213655_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=7

my_map["run_20260227_213256_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=3

my_map["run_20260227_212854_rufimelo/vulnerable_code_qwen_coder_standard_16384"]=0


# python sparse_autoencoders/feature_hypothesis.py --run_id run_20260218_134529_vulnerable_code_qwen_coder_standard_16384_10M --layer 11 --output my_hypotheses.jsonl --region us-east-2
for key in "${!my_map[@]}"; do
    value="${my_map[$key]}"
    echo "Key: $key, Value: $value"
    python -m sae_java_bug.sparse_autoencoders.feature_hypothesis --run_id "$key" --layer "$value" --output "hypotheses_layer_${value}.jsonl" --region us-east-2
done