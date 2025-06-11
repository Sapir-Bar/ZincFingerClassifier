import torch
from torch.utils.data import DataLoader
from models.model import BiLSTMClassifier
from dataset.loader import ZincFingerDataset
from train.engine import train_one_epoch, evaluate
import wandb

def run_fold(train_items, val_items, feature_dir, epochs=10, batch_size=32, lr=1e-3, device=None, fold_idx=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb for this fold
    wandb.init(
        project="ZincFingerClassification",
        name=f"fold_{fold_idx}" if fold_idx is not None else None,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "fold": fold_idx
        },
        reinit=True
    )

    train_loader = DataLoader(ZincFingerDataset(train_items, feature_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ZincFingerDataset(val_items, feature_dir), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    model = BiLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss() # Binary Cross-Entropy Loss 

    history = {
        "train_loss": [],
        "train_mcc": [],
        "val_mcc": [],
        "train_auc": [],
        "val_auc": [],
    }

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_mcc, train_auc = evaluate(model, train_loader, device, validation=False)
        val_mcc, val_auc = evaluate(model, val_loader, device, validation=True)

        history["train_loss"].append(train_loss)
        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_mcc": train_mcc,
            "val_mcc": val_mcc,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "fold": fold_idx
        })

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | "
              f"Train MCC: {train_mcc:.3f}, AUC: {train_auc:.3f} | "
              f"Val MCC: {val_mcc:.3f}, AUC: {val_auc:.3f}")
        
    wandb.finish()
    return history

from torch.utils.data import DataLoader

def custom_collate(batch):
    xs, ys, items = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), list(items)
