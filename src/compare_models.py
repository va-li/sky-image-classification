# %%
import pandas as pd
import json
from pathlib import Path
import re
from typing import Dict, Any, List
from tqdm import tqdm

# %%
IMAGE_SIZE_PATTERN = re.compile(r"Resize\(size=\[(\d+), \d+\]")

train_run_data_path = Path("../data/training-runs/").resolve()

classes = ["clouds", "rain", "dew", "clear sky", "soiling"]

results_df = []

for train_run_dir in tqdm(train_run_data_path.glob("mobilenetv3_20241220*")):
    if not train_run_dir.is_dir():
        continue

    train_run_id = train_run_dir.name
    hyperparameters: Dict[str, Any] = json.loads(
        (train_run_dir / "hyperparameters.json").read_text()
    )
    train_results = pd.read_csv(train_run_dir / "train_metrics.csv")

    if (train_run_dir / "test_metrics.json").exists():
        test_results: Dict[str, Any] = json.loads(
            (train_run_dir / "test_metrics.json").read_text()
        )

        test_mean_jaccard: float = test_results["test_mean_jaccard"]
        test_mean_precision: float = test_results["test_mean_precision"]
        test_mean_recall: float = test_results["test_mean_recall"]

        test_class_recalls: List[float] = test_results["test_class_recalls"]
        test_class_precisions: List[float] = test_results["test_class_precisions"]
        test_class_accuracies: List[float] = test_results["test_class_accuracies"]
    else:
        test_results = None

    hyperparameters["transform_train"]
    
    if isinstance(hyperparameters["transform_train"], list):
        training_augmentations = hyperparameters["transform_train"]
    elif isinstance(hyperparameters["transform_train"], str):
        training_augmentations = [t.strip() for t in hyperparameters["transform_train"].split("\n")]
    else:
        raise ValueError("Unknown type of train_transform")
    print(training_augmentations)
    color_jitter = [t for t in training_augmentations if "ColorJitter" in t]
    random_horizontal_flip = [t for t in training_augmentations if "RandomHorizontalFlip" in t]
    random_rotation = [t for t in training_augmentations if "RandomRotation" in t]
    
    if "model_image_input_size" in hyperparameters:
        image_input_size = hyperparameters["model_image_input_size"]
    else:
    
        if isinstance(hyperparameters["transform_train"], list):
            resize_transorm = [t for t in hyperparameters["transform_train"] if "Resize" in t]
            image_input_size = int(
                IMAGE_SIZE_PATTERN.findall(hyperparameters["transform_train"])[0]
            )
        elif isinstance(hyperparameters["transform_train"], str):
            image_input_size = int(
                IMAGE_SIZE_PATTERN.findall(hyperparameters["transform_train"])[0]
            )
        else:
            raise ValueError("Unknown type of train_transform")
        
    
    random_split = hyperparameters.get("random_split", None)
    learning_rate = hyperparameters.get("learning_rate", None)
    n_epochs = hyperparameters.get("n_epochs", None)
    freeze_pretrained_layers = hyperparameters.get("freeze_backbone", False)
    batch_size = hyperparameters.get("batch_size", None)
    train_set_len = len(hyperparameters["dataset_indices"]["train"])
    val_set_len = len(hyperparameters["dataset_indices"]["val"])
    test_set_len = len(hyperparameters["dataset_indices"]["test"])
    best_train_loss = train_results["train_loss"].min()
    best_val_loss = train_results["val_loss"].min()
    train_time_total_seconds = (
        train_results["train_time_seconds"].sum()
        + train_results["val_time_seconds"].sum()
    )

    index_best_val_mean_jaccard = train_results["val_mean_jaccard"].idxmax()
    best_val_mean_jaccard = train_results["val_mean_jaccard"][
        index_best_val_mean_jaccard
    ]
    best_val_mean_precision = train_results["val_mean_precision"][
        index_best_val_mean_jaccard
    ]
    best_val_mean_recall = train_results["val_mean_recall"][index_best_val_mean_jaccard]

    results_df.append(
        {
            "train_run_id": train_run_id,
            "training_augmentations": training_augmentations,
            "color_jitter": color_jitter,
            "random_horizontal_flip": random_horizontal_flip,
            "random_rotation": random_rotation,
            "image_input_size": image_input_size,
            "random_split": random_split,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "freeze_pretrained_layers": freeze_pretrained_layers,
            "batch_size": batch_size,
            "train_set_len": train_set_len,
            "val_set_len": val_set_len,
            "test_set_len": test_set_len,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "train_time_total_seconds": train_time_total_seconds,
            "best_val_mean_jaccard": best_val_mean_jaccard,
            "best_val_mean_precision": best_val_mean_precision,
            "best_val_mean_recall": best_val_mean_recall,
            "test_mean_jaccard": test_mean_jaccard if test_results else None,
            "test_mean_precision": test_mean_precision if test_results else None,
            "test_mean_recall": test_mean_recall if test_results else None,
        }
    )

results_df = pd.DataFrame(results_df)

# higher jaccard score is better
results_df.sort_values("best_val_mean_jaccard", ascending=False, inplace=True)
results_df["rank"] = list(range(1, len(results_df) + 1))

best_model = results_df.iloc[0]
# %%
results_df

# %%
print("Best model:")
print(best_model)

# %%
