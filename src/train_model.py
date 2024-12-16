import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import time
import json
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any

from dataset import SkyImageMultiLabelDataset
from model import MultiLabelClassificationMobileNetV3Large

# each training run gets its own directory to store the model, metrics and logs
training_run_timestamp = time.strftime("%Y%m%d-%H%M%S%z")
training_run_data_path = Path(
    f"/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/training-runs/mobilenetv3_{training_run_timestamp}"
)
training_run_data_path.mkdir(parents=True, exist_ok=True)

# log to a file and to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # log to stdout and to a file
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(training_run_data_path / "training.log"),
    ],
)


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
    prediction_threshold: float,
    num_classes: int,
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
    num_classes : int
        The number of classes in the dataset.

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


try:
    ###########################################################################
    #                                  Training                               #
    ###########################################################################

    # Transformations for training, validation and test sets
    transform_train = v2.Compose(
        [
            v2.ToTensor(),  # first, convert image (numpy array) to PyTorch tensor, so that further processing can be done
            v2.RandomHorizontalFlip(),  # randomly flip and rotate
            v2.RandomRotation(
                (0, 180)
            ),  # randomly rotate the image between 0 and 180 degrees
            v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),  # randomly change the brightness, contrast, saturation and hue
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_val_test = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # hyperparameters dictionary
    # all hyperparameters are stored in a dictionary and saved to a json file for reproducibility
    hyperparameters = {}

    # load the dataset
    dataset_path = Path(
        "/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/"
    )
    hyperparameters["dataset_path"] = str(dataset_path)
    dataset = SkyImageMultiLabelDataset(dataset_path)
    train_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_train)
    hyperparameters["transform_train"] = str(transform_train)
    val_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_val_test)
    test_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_val_test)
    hyperparameters["transform_val_test"] = str(transform_val_test)

    # a fixed seed for reproducibility (randomly chosen)
    SEED = 18759
    torch.manual_seed(SEED)
    hyperparameters["seed"] = SEED

    # split the dataset into training, validation and test sets
    VAL_SIZE_RATIO = 0.1
    VAL_SIZE = int(VAL_SIZE_RATIO * len(dataset))
    hyperparameters["val_size_ratio"] = VAL_SIZE_RATIO
    TEST_SIZE_RATIO = 0.1
    TEST_SIZE = int(TEST_SIZE_RATIO * len(dataset))
    hyperparameters["test_size_ratio"] = TEST_SIZE_RATIO
    TRAIN_SIZE_RATIO = 1 - VAL_SIZE_RATIO - TEST_SIZE_RATIO
    TRAIN_SIZE = len(dataset) - VAL_SIZE - TEST_SIZE
    hyperparameters["train_size_ratio"] = TRAIN_SIZE_RATIO
    RANDOM_SPLIT = False
    hyperparameters["random_split"] = RANDOM_SPLIT
    # first split the dataset into training+validation and test sets
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=RANDOM_SPLIT,
    )
    # then split the training+validation set into training and validation sets
    train_indices, val_indices = train_test_split(
        train_indices, test_size=VAL_SIZE, random_state=SEED, shuffle=RANDOM_SPLIT
    )

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # store the indices of the datasets for reproducibility
    train_test_val_indices = {
        "train": train_dataset.indices,
        "val": val_dataset.indices,
        "test": test_dataset.indices,
    }
    hyperparameters["dataset_indices"] = train_test_val_indices

    BATCH_SIZE = 32
    hyperparameters["batch_size"] = BATCH_SIZE

    # create dataloaders, shuffle the training set, but not the validation and test sets
    SUFFLE_TRAIN = True
    hyperparameters["shuffle_train"] = SUFFLE_TRAIN
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=SUFFLE_TRAIN
    )
    SUFFLE_VAL = False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SUFFLE_VAL)
    hyperparameters["shuffle_val"] = SUFFLE_VAL
    SHUFFLE_TEST = False
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TEST)
    hyperparameters["shuffle_test"] = SHUFFLE_TEST

    logging.info(
        f"Train set:\n{dataset.image_labels_df.iloc[train_dataset.indices].sum()}"
    )
    logging.info(
        f"Validation set:\n{dataset.image_labels_df.iloc[val_dataset.indices].sum()}"
    )
    logging.info(
        f"Test set:\n{dataset.image_labels_df.iloc[test_dataset.indices].sum()}"
    )

    NUM_CLASSES = len(dataset.label_names)
    hyperparameters["num_classes"] = NUM_CLASSES

    # Initialize the model
    model = MultiLabelClassificationMobileNetV3Large(num_classes=NUM_CLASSES)

    hyperparameters["model"] = "MultiLabelClassificationMobileNetV3Large"
    hyperparameters["classifier"] = str(model.classifier)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    # BCELoss is used for multi-label classification
    bce_weights = 1 / dataset.image_labels_df.iloc[train_dataset.indices].mean()
    hyperparameters["loss_function_weights"] = bce_weights.to_dict()
    criterion = nn.BCELoss(weight=torch.tensor(bce_weights.values).float().to(device))
    LEARNING_RATE = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    hyperparameters["learning_rate"] = LEARNING_RATE

    # Training loop
    N_EPOCHS = 20
    hyperparameters["n_epochs"] = N_EPOCHS
    FRZAE_LAYERS = False
    hyperparameters["freeze_layers"] = FRZAE_LAYERS

    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    train_losses = []
    train_times = []
    val_losses = []
    val_subset_accuracies = []
    val_mean_accuracies = []
    val_mean_precisions = []
    val_mean_recalls = []
    val_mean_jaccards = []
    val_times = []

    # prediction threshold above which a model output is considered a positive prediction
    PREDICTION_THRESHOLD = 0.5
    hyperparameters["prediction_threshold"] = PREDICTION_THRESHOLD

    (training_run_data_path / "hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=4)
    )

    for epoch in range(N_EPOCHS):
        # freeze all layers except the classifier
        if FRZAE_LAYERS:
            model.freeze_backbone()

        train_results = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, N_EPOCHS
        )

        train_losses.append(train_results["loss"])
        train_times.append(train_results["duration"])

        val_results = evaluate_model(
            model, criterion, val_loader, device, PREDICTION_THRESHOLD, NUM_CLASSES
        )

        val_losses.append(val_results["loss"])
        val_times.append(val_results["duration"])
        val_subset_accuracies.append(val_results["subset_accuracy"])
        val_mean_jaccards.append(val_results["mean_jaccard"])
        val_mean_precisions.append(val_results["mean_precision"])
        val_mean_recalls.append(val_results["mean_recall"])
        val_mean_accuracies.append(val_results["mean_accuracy"])

        logging.info(f"Results Epoch {epoch+1}/{N_EPOCHS}")

        for class_label, conf in enumerate(val_results["confusion_matrices"]):
            accuracy = val_results["class_accuracies"][class_label]
            recall = val_results["class_recalls"][class_label]
            precision = val_results["class_precisions"][class_label]

            logging.info(f"Label: {dataset.label_names[class_label]}")
            logging.info(
                f"Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}\n"
                + pd.DataFrame(
                    conf, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
                ).to_markdown()
            )

        # save the model if the validation loss is the best so far
        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), training_run_data_path / "best_model.pth")
            logging.info(f"New best model saved at epoch {epoch+1}")

        logging.info(f"Epoch {epoch+1} summary:")
        logging.info(
            f"Train Loss: {train_results['loss']:.4f}, Val Loss: {val_results['loss']:.4f}, Time: train {train_results['duration']:.0f}s + val {val_results['duration']:.0f}s"
        )
        logging.info(
            f"Val Mean Accuracy: {val_mean_accuracies[-1]:.4f}, Val Mean Precision: {val_mean_precisions[-1]:.4f}, Val Mean Recall: {val_mean_recalls[-1]:.4f}"
        )
        logging.info(
            f"Val Subset Accuracy: {val_subset_accuracies[-1]:.4f}, Val Mean Jaccard: {val_mean_jaccards[-1]:.4f}"
        )

        # save the metrics of the training run so far
        train_metrics = pd.DataFrame(
            {
                "train_loss": train_losses,
                "train_time_seconds": train_times,
                "val_loss": val_losses,
                "val_subset_accuracy": val_subset_accuracies,
                "val_mean_accuracy": val_mean_accuracies,
                "val_mean_jaccard": val_mean_jaccards,
                "val_mean_precision": val_mean_precisions,
                "val_mean_recall": val_mean_recalls,
                "val_time_seconds": val_times,
            },
            index=range(1, epoch + 2),
        )
        train_metrics.to_csv(
            training_run_data_path / "train_metrics.csv", index_label="epoch"
        )

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(train_metrics["train_loss"], color=color, label="Training")
        ax1.plot(
            train_metrics["val_loss"],
            color=color,
            linestyle="dashed",
            label="Validation",
        )
        ax1.tick_params(axis="y", labelcolor=color)
        # legend in upper left corner
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Subset Accuracy", color=color)
        ax2.plot(train_metrics["val_subset_accuracy"], color=color, label="Validation")
        ax2.tick_params(axis="y", labelcolor=color)
        # legend in upper right corner
        ax2.legend(loc="upper right")

        # set the x-axis to be the epoch number
        ax1.set_xticks(train_metrics.index)
        ax1.set_xticklabels(train_metrics.index)

        fig.suptitle(f"{training_run_data_path.name} - Training Metrics")
        fig.tight_layout()
        # save the plot
        plt.savefig(training_run_data_path / "metrics_plot.png")

        # plot mean jaccard, mean accuracy, mean precision and mean recall
        fig, ax = plt.subplots()
        ax.plot(train_metrics["val_mean_jaccard"], label="Mean Jaccard")
        ax.plot(train_metrics["val_mean_accuracy"], label="Mean Accuracy")
        ax.plot(train_metrics["val_mean_precision"], label="Mean Precision")
        ax.plot(train_metrics["val_mean_recall"], label="Mean Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        ax.set_xticks(train_metrics.index)
        ax.set_xticklabels(train_metrics.index)
        fig.suptitle(f"{training_run_data_path.name} - Validation Metrics")
        fig.tight_layout()
        plt.savefig(training_run_data_path / "validation_metrics_plot.png")

    logging.info("Training complete")

    ###########################################################################
    #                        Evaluation on test set                           #
    ###########################################################################

    # load the best model of the training run
    model = MultiLabelClassificationMobileNetV3Large(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(
            training_run_data_path / "best_model.pth",
            map_location=torch.device(device),
        )
    )
    model = model.to(device)

    test_results = evaluate_model(
        model, criterion, test_loader, device, PREDICTION_THRESHOLD, NUM_CLASSES
    )

    logging.info("Test set results:")
    logging.info(
        f"Test Loss: {test_results['loss']:.4f}, Time: {test_results['duration']:.0f}s"
    )
    logging.info(
        f"Test Mean Accuracy: {test_results['mean_accuracy']:.4f}, Test Mean Precision: {test_results['mean_precision']:.4f}, Test Mean Recall: {test_results['mean_recall']:.4f}"
    )
    logging.info(
        f"Test Subset Accuracy: {test_results['subset_accuracy']:.4f}, Test Mean Jaccard: {test_results['mean_jaccard']:.4f}"
    )

    for class_label, conf in enumerate(test_results["confusion_matrices"]):
        accuracy = test_results["class_accuracies"][class_label]
        recall = test_results["class_recalls"][class_label]
        precision = test_results["class_precisions"][class_label]

        logging.info(f"Label: {dataset.label_names[class_label]}")
        logging.info(
            f"Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}\n"
            + pd.DataFrame(
                conf, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
            ).to_markdown()
        )

    # save the test prediciton results
    test_results_df = pd.DataFrame(
        {
            "dataset_indices": test_dataset.indices,
            "true_labels": test_results["true_labels"],
            "predicted_labels": test_results["predicted_labels"],
        },
        index=dataset.image_labels_df.index[test_dataset.indices].index.tolist(),
    )

    test_results_df.to_csv(training_run_data_path / "test_results.csv")

    # save the test metrics
    test_metrics = {
        "test_loss": test_results["loss"],
        "test_time_seconds": test_results["duration"],
        "test_subset_accuracy": test_results["subset_accuracy"],
        "test_mean_accuracy": test_results["mean_accuracy"],
        "test_mean_jaccard": test_results["mean_jaccard"],
        "test_mean_precision": test_results["mean_precision"],
        "test_mean_recall": test_results["mean_recall"],
        "test_confusion_matrices": list(
            map(lambda conf: conf.tolist(), test_results["confusion_matrices"])
        ),
        "test_class_accuracies": test_results["class_accuracies"].tolist(),
        "test_class_recalls": test_results["class_recalls"].tolist(),
        "test_class_precisions": test_results["class_precisions"].tolist(),
    }

    (training_run_data_path / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=4)
    )

    logging.info("Evaluations complete")

except Exception as e:
    # print the full traceback to the log file
    logging.exception("Exception occurred")

    # re-raise the exception to see the full traceback in the notebook
    raise e
