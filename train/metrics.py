from sklearn.metrics import matthews_corrcoef, roc_auc_score

def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = float('nan')  # Handle case where AUC cannot be computed (e.g., all labels are the same)
        
    return mcc, auc


