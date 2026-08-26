import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

def calculate_metrics(y_true, y_prob, y_pred, num_classes):
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['bacc'] = balanced_accuracy_score(y_true, y_pred)
    
    if num_classes == 2:
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        # y_prob should be probabilities for positive class (class 1)
        metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
    else:
        # Multi-class
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        # PR-AUC for multi-class requires one-vs-rest averaging or similar, scikit-learn doesn't support directly in average_precision_score
        # Calculate macro PR-AUC manually
        y_true_onehot = np.eye(num_classes)[y_true]
        metrics['pr_auc'] = average_precision_score(y_true_onehot, y_prob, average='macro')
        
    return metrics
