import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_auc_per_protein(truth_df, pred_df, name="Predictor"):
    # Merge ground truth and predictions according to prot_name_id, zf_index columns
    merged = pd.merge(
        truth_df[["prot_name_id", "zf_index", "connection_label"]],
        pred_df[["prot_name_id", "zf_index", "pred_label_value"]],
        on=["prot_name_id", "zf_index"],
        how="inner"
    )

    # Group by prot_name_id
    grouped = merged.groupby("prot_name_id")
    aucs = []

    for prot_id, group in grouped:
        y_true = group["connection_label"].astype(int)
        y_score = group["pred_label_value"].astype(float)

        # Check if there are at least two classes in y_true
        if len(set(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_score)
        aucs.append({"prot_name_id": prot_id, "AUC": auc})

    auc_df = pd.DataFrame(aucs)

    # Box plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=auc_df["AUC"])
    plt.title(f"Distribution of AUC-ROC Scores Across Zinc Finger Proteins")
    plt.ylabel("AUC-ROC")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # statistics
    mean_auc = auc_df["AUC"].mean()
    std_auc = auc_df["AUC"].std()
    print(f"{name} - Mean AUC: {mean_auc:.3f}, Std: {std_auc:.3f}")

    return auc_df

def evaluate_mcc_per_protein(truth_df, pred_df, name="Predictor"):
    # Merge ground truth and predictions according to prot_name_id, zf_index columns
    merged = pd.merge(
        truth_df[["prot_name_id", "zf_index", "connection_label"]],
        pred_df[["prot_name_id", "zf_index", "pred_label"]],
        on=["prot_name_id", "zf_index"],
        how="inner"
    )

    # Group by prot_name_id
    grouped = merged.groupby("prot_name_id")
    mccs = []

    for prot_id, group in grouped:
        y_true = group["connection_label"].astype(int)
        y_pred = group["pred_label"].astype(int)

        # Skip proteins with only one class
        if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
            continue

        mcc = matthews_corrcoef(y_true, y_pred)
        mccs.append({"prot_name_id": prot_id, "MCC": mcc})

    mcc_df = pd.DataFrame(mccs)

    # Boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(y=mcc_df["MCC"])
    plt.title(f"Distribution of MCC Scores Across Zinc Finger Proteins")
    plt.ylabel("Matthews Correlation Coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # statistics
    mean_mcc = mcc_df["MCC"].mean()
    std_mcc = mcc_df["MCC"].std()
    print(f"{name} - Mean MCC: {mean_mcc:.3f}, Std: {std_mcc:.3f}")

    return mcc_df