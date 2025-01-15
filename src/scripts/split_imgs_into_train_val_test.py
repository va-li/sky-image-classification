# %%
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from pathlib import Path
import json
import shutil

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset import SkyImageMultiLabelDataset

torch.backends.cudnn.deterministic = True

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

# load the test dataset
dataset_path = Path("../data/").resolve()
hyperparameters_train = json.loads(
    (training_run_data_path / "hyperparameters.json").read_text()
)

dataset = SkyImageMultiLabelDataset(dataset_path)
test_indices = hyperparameters_train["dataset_indices"]["test"]
test_dataset = torch.utils.data.Subset(dataset, test_indices)
train_indices = hyperparameters_train["dataset_indices"]["train"]
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_indices = hyperparameters_train["dataset_indices"]["val"]
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# create train, val, test folders
train_folder = Path("../data/sky-images/train/").resolve()
val_folder = Path("../data/sky-images/val/").resolve()
test_folder = Path("../data/sky-images/test/").resolve()
    
train_file_paths = [Path(p) for p in dataset.image_metadata_df.iloc[train_indices].index]
val_file_paths = [Path(p) for p in dataset.image_metadata_df.iloc[val_indices].index]
test_file_paths = [Path(p) for p in dataset.image_metadata_df.iloc[test_indices].index]

assert set(train_file_paths).isdisjoint(val_file_paths), "Train and val sets are not disjoint."
assert set(train_file_paths).isdisjoint(test_file_paths), "Train and test sets are not disjoint."
assert set(val_file_paths).isdisjoint(test_file_paths), "Val and test sets are not disjoint."

print(f"Copying {len(train_indices)} images to {train_folder}")
print(f"Copying {len(val_indices)} images to {val_folder}")
print(f"Copying {len(test_indices)} images to {test_folder}")

yN = input("Do you want to continue? (y/N): ")
if yN.lower() != "y":
    print("Aborted.")
    sys.exit(0)

if not train_folder.exists():
    train_folder.mkdir()
if not val_folder.exists():
    val_folder.mkdir()
if not test_folder.exists():
    test_folder.mkdir()
    
# create files for train, val, test sets with the file paths of the corresponding images
(train_folder / "train_image_files.txt").write_text("\n".join([str(p) for p in train_file_paths]))
(val_folder / "val_image_files.txt").write_text("\n".join([str(p) for p in val_file_paths]))
(test_folder / "test_image_files.txt").write_text("\n".join([str(p) for p in test_file_paths]))    

# copy images to train, val, test folders
for rel_file_path in dataset.image_metadata_df.iloc[train_indices].index:
    original_file_path = dataset_path / rel_file_path
    target_file_path = train_folder / Path(rel_file_path).name
    # print(f"{original_file_path}\n-> {target_file_path}")
    shutil.copy(original_file_path, target_file_path)
    
print("Train images copied.")
    
for rel_file_path in dataset.image_metadata_df.iloc[val_indices].index:
    original_file_path = dataset_path / rel_file_path
    target_file_path = val_folder / Path(rel_file_path).name
    # print(f"{original_file_path}\n-> {target_file_path}")
    shutil.copy(original_file_path, target_file_path)
    
print("Val images copied.")
    
for rel_file_path in dataset.image_metadata_df.iloc[test_indices].index:
    original_file_path = dataset_path / rel_file_path
    target_file_path = test_folder / Path(rel_file_path).name
    # print(f"{original_file_path}\n-> {target_file_path}")
    shutil.copy(original_file_path, target_file_path)

print("Test images copied.")
