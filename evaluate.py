    
import matplotlib.pyplot as plt
import numpy as np

def plot_evaluation_results(val_aucs, val_mccs):
    """
    Plots the validation AUC and MCC for each fold as box plots.

    Parameters:
    - val_aucs: List of AUC values for each fold.
    - val_mccs: List of MCC values for each fold.
    """

    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)
    mean_mcc = np.mean(val_mccs)
    std_mcc = np.std(val_mccs)

    print(f"Average AUC across folds: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"Average MCC across folds: {mean_mcc:.3f} ± {std_mcc:.3f}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.boxplot(val_aucs, vert=True, patch_artist=True)
    plt.title("Validation AUC per Fold")
    plt.ylabel("AUC")
    plt.xticks([1], ["AUC"])

    plt.subplot(1,2,2)
    plt.boxplot(val_mccs, vert=True, patch_artist=True)
    plt.title("Validation MCC per Fold")
    plt.ylabel("MCC")
    plt.xticks([1], ["MCC"])

    plt.tight_layout()
    plt.savefig("results/validation_results.png")