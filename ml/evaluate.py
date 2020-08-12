from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import torch
from torch import tensor
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report

from .models import RatioModel
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def evaluate_ratio_model(
    model,
    xs=None,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    n_xs = len(xs)
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)
    with torch.no_grad():
        model.eval()

        r_hat, s_hat  = model(xs)
        # Copy back tensors to CPU
        if run_on_gpu:
            r_hat = r_hat.cpu()
            s_hat = s_hat.cpu()

        # Get data and return
        r_hat = r_hat.detach().numpy().flatten()
        s_hat = s_hat.detach().numpy().flatten()
    return r_hat, s_hat

def evaluate_performance_model(
    model,
    xs,
    ys,
    run_on_gpu=True,
    double_precision=False,
    return_grad_x=False,
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

    # Prepare data
    n_xs = len(xs)
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)

    with torch.no_grad():
        model.eval()

        _, y_test_pred  = model(xs)
        y_pred_tag = torch.round(y_test_pred)
        print("accuracy ",accuracy_score(y_pred_tag,ys))
        print("confusion matrix ",confusion_matrix(ys, y_pred_tag))
        print(classification_report(ys, y_pred_tag))
        fpr, tpr, auc_thresholds = roc_curve(ys, y_pred_tag)
        print("auc ", auc(fpr, tpr)) 
        plot_roc_curve(fpr, tpr, 'test')
        plt.savefig('plots/rocClassifier.png')
        plt.clf()

def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
