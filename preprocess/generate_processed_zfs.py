import os
import json
import numpy as np
import pandas as pd

def load_feature_matrix(prot_name_id, feature_dir):
    path = os.path.join(feature_dir, f"{prot_name_id}.fea")
    return np.loadtxt(path)  # shape: (L, 1280)

def extract_zf_representation(matrix, start, end, context=20):
    L, dim = matrix.shape
    zf_center_start = max(start - context, 0)
    zf_center_end = min(end + context, L)

    submatrix = matrix[zf_center_start:zf_center_end]
    missing = 52 - submatrix.shape[0]

    if missing > 0:
        pad_top = (context - (start - zf_center_start))
        pad_bottom = missing - pad_top
        padding = ((pad_top, pad_bottom), (0, 0))
        submatrix = np.pad(submatrix, padding, mode='constant', constant_values=0)

    return submatrix  # shape: (52, 1280)

def generate_processed_zfs(csv_path, feature_dir, out_dir):
    df = pd.read_csv(csv_path, delim_whitespace=True)
    os.makedirs(out_dir, exist_ok=True)

    for i, row in df.iterrows():
        prot_id = row['prot_name_id']
        zf_idx = row['zf_index']
        start = int(row['zf_indx_start'])
        end = int(row['zf_indx_end'])
        label = int(float(row['connection_label']))

        feature_matrix = load_feature_matrix(prot_id, feature_dir)
        zf_matrix = extract_zf_representation(feature_matrix, start, end)

        # Save matrix
        np.save(os.path.join(out_dir, f"{prot_id}_zf{zf_idx}.npy"), zf_matrix)

        # Save metadata
        meta = {
            'label': label,
            'prot_name_id': prot_id,
            'zf_index': int(zf_idx),
            'zf_start': start,
            'zf_end': end
        }
        with open(os.path.join(out_dir, f"{prot_id}_zf{zf_idx}.json"), "w") as f:
            json.dump(meta, f)

if __name__ == "__main__":
    csv_path = "data/zf_data_df.csv"
    feature_dir = "data/feature_matrices"
    out_dir = "data/ProcessedZFs"
    generate_processed_zfs(csv_path, feature_dir, out_dir)
