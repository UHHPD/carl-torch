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
<<<<<<< HEAD
=======
    batch_size = 10000
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
):
    # CPU or GPU?
    run_on_gpu = run_on_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if run_on_gpu else "cpu")
    dtype = torch.double if double_precision else torch.float

<<<<<<< HEAD
    # Prepare data
    n_xs = len(xs)
    xs = torch.stack([tensor(i) for i in xs])

    model = model.to(device, dtype)
    xs = xs.to(device, dtype)
    with torch.no_grad():
        model.eval()

        r_hat, s_hat  = model(xs)
        # Do we need this as ml/models.py::forward() defined implicitely that the output of the network is:
        #      s_hat = torch.sigmoid(s_hat)  where s_hat at this point is the network last layer
        #      r_hat = (1-s_hat) / s_hat = p_{1}(x) / p_{0}(x)
        #s_hat = torch.sigmoid(s_hat) 
        # Copy back tensors to CPU
        if run_on_gpu:
            r_hat = r_hat.cpu()
            s_hat = s_hat.cpu()

        # Get data and return
        r_hat = r_hat.detach().numpy().flatten()
        s_hat = s_hat.detach().numpy().flatten()
=======
    model = model.to(device, dtype)
    r_hat_list = []
    s_hat_list = []

    with torch.no_grad():
        model.eval()

        # Loop through batches
        for i in range(0, len(xs), batch_size):
            batch_xs = xs[i:i + batch_size]
            batch_xs = torch.stack([tensor(i) for i in batch_xs]).to(device, dtype)

            # Model inference
            batch_r_hat, batch_s_hat = model(batch_xs)

            # Copy back tensors to CPU
            if run_on_gpu:
                batch_r_hat = batch_r_hat.cpu()
                batch_s_hat = batch_s_hat.cpu()

            # Convert to numpy and flatten for consistent shape
            r_hat_list.append(batch_r_hat.detach().numpy().flatten())
            s_hat_list.append(batch_s_hat.detach().numpy().flatten())

    # Concatenate all batches into a single array for r_hat and s_hat
    r_hat = np.concatenate(r_hat_list)
    s_hat = np.concatenate(s_hat_list)
    
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
    return r_hat, s_hat

def evaluate_performance_model(
    model,
    xs,
    ys,
    run_on_gpu=True,
    double_precision=False,
<<<<<<< HEAD
=======
    batch_size = 10000,
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)
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
<<<<<<< HEAD

    with torch.no_grad():
        model.eval()

        _, logit  = model(xs)
        probs = torch.sigmoid(logit)
        y_pred = torch.round(probs).cpu()
        print("confusion matrix ",confusion_matrix(ys, y_pred))
        print(classification_report(ys, y_pred))
        fpr, tpr, auc_thresholds = roc_curve(ys, y_pred)
=======
    ys = torch.tensor(ys).to(device, dtype)  # Convert ys to tensor and send to the same device

    model.eval()  # Set the model to evaluation mode

    # Initialize lists to accumulate the predictions and true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Loop over the dataset in batches
        for i in range(0, n_xs, batch_size):
            batch_xs = xs[i:i + batch_size]  # Get a batch of input features
            batch_ys = ys[i:i + batch_size]  # Get the corresponding true labels

            # Forward pass on the batch
            _, logit = model(batch_xs)
            probs = torch.sigmoid(logit)  # Apply sigmoid to get probabilities
            y_pred = torch.round(probs)  # Round to get binary predictions
            
            # Append predictions and true labels to the lists
            all_preds.append(y_pred)  # Move to CPU and convert to numpy
            all_labels.append(batch_ys)  # Do the same for labels

        # Concatenate all batches' results into one array
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Now that we have all predictions and true labels, compute the performance metrics
        print("Confusion Matrix:",confusion_matrix(all_labels, all_preds))  # Confusion matrix
        print(classification_report(all_labels, all_preds))  # Classification report
        fpr, tpr, auc_thresholds = roc_curve(all_labels, all_preds)  # ROC curve
>>>>>>> d4a15a9 (Initial commit for CARL-TORCH)

def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')
