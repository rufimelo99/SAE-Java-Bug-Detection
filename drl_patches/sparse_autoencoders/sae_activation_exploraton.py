import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

def read_jsonl_file(jsonl_path):
    with open(jsonl_path, "r") as f:
        for line in f:
            yield json.loads(line)

def get_silhouette_scores(X_train, k_range):
    sil = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k).fit(X_train)
        labels = kmeans.labels_
        sil.append(silhouette_score(X_train, labels, metric='euclidean'))
    return sil

folder_path_0 = "gemma2_defects4j"
folder_path_1 = "gemma2_gbug-java"
folder_path_2 = "gemma2_humaneval"




def get_avg_silhouette_scores(folder, k, max_layers=24):

    results = []
    layer_idx = 0


    for folder in tqdm(folders):
        dirpath = os.path.join(folder_path, folder)
        if not os.path.isdir(dirpath):
            continue

        filenames = os.listdir(dirpath)

        if "layer" not in dirpath:
            continue
        # Initialize paths
        jsonl_path_diff = jsonl_path_safe = jsonl_path_vuln = None
        for filename in filenames:
            if filename == "feature_importance_diff.jsonl":
                jsonl_path_diff = os.path.join(dirpath, filename)
            elif filename == "feature_importance_safe.jsonl":
                jsonl_path_safe = os.path.join(dirpath, filename)
            elif filename == "feature_importance_vuln.jsonl":
                jsonl_path_vuln = os.path.join(dirpath, filename)

        # Skip if any required file is missing
        if not (jsonl_path_safe and jsonl_path_vuln):
            continue

        # Read and clean data
        safe_df = pd.DataFrame(read_jsonl_file(jsonl_path_safe))
        vuln_df = pd.DataFrame(read_jsonl_file(jsonl_path_vuln))

        safe_df.drop(columns=["labels", "model", "plot_type"], inplace=True)
        vuln_df.drop(columns=["labels", "model", "plot_type"], inplace=True)

        X_safe = safe_df["values"].tolist()
        X_vuln = vuln_df["values"].tolist()

        # Get scores
        sil_safe = get_silhouette_scores(X_safe, k_range)
        sil_vuln = get_silhouette_scores(X_vuln, k_range)


        results.append({
            "layer": layer_idx,
            "sil_safe": sil_safe,
            "sil_vuln": sil_vuln
        })

        layer_idx += 1
        if layer_idx >= max_layers:
            break

    return results


k_range = []
k_range.extend(list(np.arange(2, 10, 1)))
k_range.extend(list(np.arange(10, 101, 5)))
print(k_range)


results = []
for folder_path in tqdm([folder_path_1, folder_path_2, folder_path_0]):
    folders = os.listdir(folder_path)
    folders.sort(key=lambda x: int(x.split("layer")[1]))
    results.extend(get_avg_silhouette_scores(folder_path, k_range))
    print(len(results)) 

with open("silhouette_scores_gemma2.json", "w") as f:
    json.dump(results, f)

# with open("silhouette_scores.json", "r") as f:
#     results = json.load(f)