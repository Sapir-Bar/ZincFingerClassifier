import torch
from torch.utils.data import DataLoader
from models.model import BiLSTMClassifier
from dataset.loader import ZincFingerDataset
from train.engine import train_one_epoch, evaluate

def run_fold(train_items, val_items, feature_dir, epochs=10, batch_size=32, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(ZincFingerDataset(train_items, feature_dir), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ZincFingerDataset(val_items, feature_dir), batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    model = BiLSTMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss() # Binary Cross-Entropy Loss 

    history = {
        "train_mcc": [],
        "val_mcc": [],
        "train_auc": [],
        "val_auc": [],
    }

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_mcc, train_auc = evaluate(model, train_loader, device, validation=False)
        val_mcc, val_auc = evaluate(model, val_loader, device, validation=True)

        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | "
              f"Train MCC: {train_mcc:.3f}, AUC: {train_auc:.3f} | "
              f"Val MCC: {val_mcc:.3f}, AUC: {val_auc:.3f}")

    return history

from torch.utils.data import DataLoader

def custom_collate(batch):
    xs, ys, items = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32), list(items)
