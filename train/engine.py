import torch
import numpy as np
import state
from train.metrics import compute_metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y, item in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # loss.item() returns the average value of the loss for this batch
        # x.size(0) is the batch size, so we multiply the loss by the batch size to get the total loss for this batch
        total_loss += loss.item() * x.size(0) 

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, validation=False):
    model.eval()
    all_preds = []
    all_labels = []
    all_items = []

    with torch.no_grad():
        for x, y, item in loader:
            x = x.to(device)
            output = model(x)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(y.numpy())
            all_items.extend(item) 

    if validation:
        update_predictions(all_items, all_preds)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mcc, auc = compute_metrics(all_labels, all_preds)
    return mcc, auc

def update_predictions(pred_items, pred_values):
    for item, pred in zip(pred_items, pred_values):
        prot_id = item["prot_name_id"]
        zf_idx = item["zf_index"]
        pred_value = float(pred)
        pred_label = float(pred_value >= 0.5)

        df = state.df 

        row_index = df.index[(df["prot_name_id"] == prot_id) & (df["zf_index"] == zf_idx)]
        if len(row_index) == 1:
            df.at[row_index[0], "pred_label_value"] = pred_value
            df.at[row_index[0], "pred_label"] = pred_label