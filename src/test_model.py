# %%
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging

from dataset import SkyImageMultiLabelDataset
from model import MultiLabelClassificationMobileNetV3Large
from utils import evaluate_model

training_runs_path = Path("../data/training-runs/").resolve()
available_training_runs = sorted(
    [p.name for p in training_runs_path.iterdir() if p.is_dir()]
)
print("Available training runs:")
print("\n".join(available_training_runs))

selected_training_run = input(
    f"Write a training run name or press Enter to use the latest ({available_training_runs[-1]}): "
)

training_run_data_path = Path(
    "../data/training-runs/" + (selected_training_run or available_training_runs[-1])
).resolve()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # log to stdout and to a file
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(training_run_data_path / "test.log", mode="w"),
    ],
)

# 1. load data
# 2. load model
# 3. evaluate model on test data
# 4. save evaluation results

# 1. load data
# %%
try:
    # load the test dataset
    dataset_path = Path("../data/").resolve()
    hyperparameters_train = json.loads(
        (training_run_data_path / "hyperparameters.json").read_text()
    )

    transform_test_test = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize((448, 448), interpolation=v2.InterpolationMode.BICUBIC),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_test_test)
    test_indices = hyperparameters_train["dataset_indices"]["test"]
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    SHUFFLE_TEST = False
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparameters_train["batch_size"], shuffle=False
    )

    device = "cpu"

    logging.info(
        f"Test set:\n{dataset.image_labels_df.iloc[test_dataset.indices].sum()}"
    )

    NUM_CLASSES = hyperparameters_train["num_classes"]

    # load the model
    model = MultiLabelClassificationMobileNetV3Large(num_classes=NUM_CLASSES, image_input_size=hyperparameters_train["model_image_input_size"])
    try:
        model.load_state_dict(
            torch.load(
                training_run_data_path / "best_model.pth",
                map_location=torch.device(device),
                weights_only=True,
            )
        )
    except FileNotFoundError:
        print("No pretrained weights found. Model will use random initialization.")

    PREDICTION_THRESHOLD = hyperparameters_train["prediction_threshold"]

    criterion = nn.BCELoss(
        weight=torch.tensor(
            list(hyperparameters_train["loss_function_weights"].values())
        )
        .float()
        .to(device)
    )

    test_results = evaluate_model(
        model, criterion, test_loader, device, PREDICTION_THRESHOLD
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
    test_predictions = {
        "image_paths": dataset.image_labels_df.index[test_dataset.indices].tolist(),
        "dataset_indices": test_dataset.indices,
        "true_labels": test_results["true_labels"].astype(int).tolist(),
        "predicted_labels": test_results["predicted_labels"].astype(int).tolist(),
    }

    with open(training_run_data_path / "test_predictions.json", "w") as f:
        json.dump(test_predictions, f, indent=4)

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
        "test_class_accuracies": test_results["class_accuracies"],
        "test_class_recalls": test_results["class_recalls"],
        "test_class_precisions": test_results["class_precisions"],
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
