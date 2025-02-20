for i in {8..11}; do
  python sae_exploration.py --csv_path artifacts/defects4j.csv --sae_id blocks.$i.hook_resid_pre --cache_component hook_sae_acts_post --output_dir defects4j/layer$i
  python vulnerability_detection_features.py --dir-path defects4j/layer$i/
done