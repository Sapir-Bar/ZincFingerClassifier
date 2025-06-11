import os
import subprocess
import numpy as np
import pandas as pd
import json
from dataset.folds import load_metadata, split_folds
from experiments.run_fold import run_fold
import state
from evaluate import plot_evaluation_results

def main():

    # Generate pairs of fearute matrix and metadata for each ZF
    processed_dir = "data/ProcessedZFs"
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        subprocess.run(["python", "preprocess/generate_processed_zfs.py"], check=True)

    state.df = pd.read_csv("data/zf_data_df.csv", delim_whitespace=True)
    state.df["pred_label_value"] = None
    state.df["pred_label"] = None

    meta = load_metadata("data/ProcessedZFs")
    folds = split_folds(meta, n_splits=10)
    feature_dir = "data/ProcessedZFs"

    all_histories = []
    val_aucs = []
    val_mccs = []

    for fold_idx, (train_items, val_items) in enumerate(folds):
        print(f"\n--- Running Fold {fold_idx+1}/10 ---")
        history = run_fold(train_items, val_items, feature_dir, epochs=20, batch_size=32, fold_idx=fold_idx+1)
        all_histories.append(history)
        val_aucs.append(history["val_auc"][-1])
        val_mccs.append(history["val_mcc"][-1])

        # Optional: Save intermediate results
        with open(f"results/fold_{fold_idx+1}_history.json", "w") as f:
            json.dump(history, f)

    # Save all results in a single .json file
    with open("results/all_folds_history.json", "w") as f:
        json.dump(all_histories, f)

    # Save the final state DataFrame
    state.df.to_csv("results/zf_pred_df.csv", index=False)
    plot_evaluation_results(val_aucs, val_mccs)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()
