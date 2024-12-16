import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, jaccard_score
from tqdm import tqdm
import time
from typing import Dict, Any

from model import MultiLabelClassificationMobileNetV3Large

def train_one_epoch(
    model: MultiLabelClassificationMobileNetV3Large,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device | str,
    current_epoch: int,
    num_epochs: int,
) -> Dict[str, float]:
    """Train the model for one epoch and return the loss and duration of the training epoch.

    Parameters
    ----------
    model : MultiLabelClassificationMobileNetV3Large
        The custom model to train.
    criterion : nn.Module
        The loss function to use for training, for this multi-label classification task, Binary Cross Entropy Loss should be used.
    optimizer : optim.Optimizer
        The optimizer to use for training, e.g., Adam or SGD.
    dataloader : DataLoader
        The DataLoader for the training dataset.
    device : torch.device | str
        The device to use for training, either 'cuda' for GPU or 'cpu' for CPU.
    current_epoch : int
        The current epoch number (0-based).
    num_epochs : int
        The total number of epochs to train.

    Returns
    -------
    Dict[str, float]
        A dictionary with the loss (`loss`) and duration in seconds (`duration`) of the training epoch.
    """

    train_start = time.time()

    model.train()

    running_loss = 0.0

    for inputs, labels in tqdm(
        # current_epoch is 0-based, so add 1 to the epoch number
        dataloader,
        desc=f"Training Epoch {current_epoch+1}/{num_epochs}",
    ):
        # move the inputs and labels to the same device as the model (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device).float()

        # calculate the model outputs, backpropagate the loss and update the model parameters
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    train_duration = time.time() - train_start

    return {"loss": epoch_loss, "duration": train_duration}


def evaluate_model(
    model: MultiLabelClassificationMobileNetV3Large,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str,
    prediction_threshold: float
) -> Dict[str, Any]:
    """Evaluate the model on a dataset and return the metrics and predictions.

    The metrics returned are the loss, subset accuracy, mean jaccard score, mean precision, mean recall and mean accuracy.
    Additionally, the predicted labels, true labels and confusion matrices for each class are returned.

    Parameters
    ----------
    model : MultiLabelClassificationMobileNetV3Large
        The custom model to evaluate.
    criterion : nn.Module
        The loss function to use for evaluation, for this multi-label classification task, Binary Cross Entropy Loss should be used.
    dataloader : DataLoader
        The DataLoader for the dataset to evaluate the model on.
    device : torch.device | str
        The device to use for evaluation, either 'cuda' for GPU or 'cpu' for CPU.
    prediction_threshold : float
        The threshold above which a model output is considered a positive prediction.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the evaluation metrics.
        Supported keys and value types:
        - scalars: loss (`loss`), duration in seconds (`duration`), subset accuracy (`subset_accuracy`), mean jaccard score (`mean_jaccard`),
            mean precision (`mean_precision`), mean recall (`mean_recall`), mean accuracy (`mean_accuracy`)
        - lists/numpy arrays: predicted labels for each observation(`predicted_labels`), true labels for each observation (`true_labels`),
            confusion matrices for each class (`confusion_matrices`), class accuracies for each class (`class_accuracies`),
            class recalls for each class (`class_recalls`) and class precisions for each class (`class_precisions`)
    """

    val_start = time.time()

    model.eval()

    loss = 0.0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating model"):
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item() * inputs.size(0)

            predicted_labels = outputs > prediction_threshold

            all_predicted_labels.extend(predicted_labels.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    duration = time.time() - val_start
    
    all_predicted_labels = np.array(all_predicted_labels).astype(int)
    all_true_labels = np.array(all_true_labels).astype(int)

    loss /= len(dataloader.dataset)

    # calculate the subset accuracy and jaccard score for the given dataset
    val_subset_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    val_mean_jaccard = jaccard_score(
        all_true_labels, all_predicted_labels, average="samples"
    )  # calculated for each sample and then averaged

    # calculate confusion matrices (one-vs-rest) for each class
    confusion_matrices = multilabel_confusion_matrix(
        all_true_labels, all_predicted_labels
    )

    # calculate the accuracy, recall and precision for each class
    # and the mean accuracy, recall and precision for the given dataset

    # list of class accuracies, recalls and precisions
    # ordered by the class index
    class_accuracies = []
    class_recalls = []
    class_precisions = []
    for conf in confusion_matrices:

        # calculate the true positives, true negatives, false positives and false negatives
        tp = conf[1, 1]
        tn = conf[0, 0]
        fp = conf[0, 1]
        fn = conf[1, 0]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        class_accuracies.append(accuracy)
        recall = tp / (tp + fn)
        class_recalls.append(recall)
        precision = tp / (tp + fp)
        class_precisions.append(precision)

    mean_precision = np.nanmean(class_precisions)
    mean_recall = np.nanmean(class_recalls)
    mean_accuracy = np.nanmean(class_accuracies)

    return {
        "loss": loss,
        "duration": duration,
        "subset_accuracy": val_subset_accuracy,
        "mean_jaccard": val_mean_jaccard,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_accuracy": mean_accuracy,
        "predicted_labels": all_predicted_labels,
        "true_labels": all_true_labels,
        "confusion_matrices": confusion_matrices,
        "class_accuracies": class_accuracies,
        "class_recalls": class_recalls,
        "class_precisions": class_precisions,
    }
