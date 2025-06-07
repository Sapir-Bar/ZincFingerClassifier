import os
import json
from collections import defaultdict
from sklearn.model_selection import KFold

def load_metadata(json_dir):
    items = []
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname), "r") as f:
                meta = json.load(f)
                meta["json_path"] = os.path.join(json_dir, fname)
                meta["npy_path"] = fname.replace(".json", ".npy")
                items.append(meta)
    return items

def group_by_protein(meta_items):
    prot_to_items = defaultdict(list)
    # Create a dictionary of type {prot_name_id: [ZF1, ZF2, ...]}
    for item in meta_items:
        prot_to_items[item["prot_name_id"]].append(item)
    return prot_to_items

def split_folds(meta_items, n_splits=10, seed=42):
    prot_to_items = group_by_protein(meta_items)
    prot_ids = list(prot_to_items.keys())

    # Split the 157 C2H2-ZF proteins into 10 folds as reported in the paper
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    # kf.split() returns for each fold 2 lists: train indices and test indices
    for train_idx, test_idx in kf.split(prot_ids):
        # Extract the protein IDs for the current fold
        train_prots = [prot_ids[i] for i in train_idx]
        test_prots = [prot_ids[i] for i in test_idx]

        # Extract the ZFs for the current fold
        train_set = []
        for p in train_prots:
            for item in prot_to_items[p]:
                train_set.append(item)

        test_set = []
        for p in test_prots:
            for item in prot_to_items[p]:
                test_set.append(item)
                
        folds.append((train_set, test_set))

    return folds
