from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io


class SkyImageMultiLabelDataset(Dataset):
    """Sky Image Multi-Label Dataset

    The dataset contains hemispherical images of the sky and for each image labels (at least one). Images without labels in the dataset directory are ignored.
    It can be filtered by label names with the `get_integer_indices_for_labels` and `get_integer_indices_for_exclusive_labels` methods.
    The labels are in ImageNet format, i.e., a list of labels for each image and a list of label names.
    """

    def __init__(
        self,
        root_dir: Path,
        image_labels_file: str = "default.txt",
        label_names_file: str = "synsets.txt",
        transform: transforms.Compose | None = None,
    ):
        """Initialize the Sky Image Multi-Label Dataset

        Parameters
        ----------
        root_dir : Path
            The root directory of the dataset. The image_labels_file and label_names_file should be in this directory.
        image_labels_file : str
            The name of the file containing the image labels in ImageNet format (e.g., 'image_1.jpg 0 2'), by default 'default.txt'
        label_names_file : str
            The name of the file containing the label names in ImageNet format (e.g., 'clear sky', one label per line), by default 'synsets.txt'
        transform : torchvision.transforms.Compose
            The transformation to apply to the images, by default None
        """

        self.root_dir = root_dir
        self.image_labels_file_path = root_dir / image_labels_file
        self.label_names_file_path = root_dir / label_names_file
        self.transform = transform

        # read the label names
        with open(self.label_names_file_path, "r") as f:
            # line number corresponds to the label number, first line is label 0
            label_names = [name.strip() for i, name in enumerate(f.readlines())]

        self.label_names = label_names

        # read the image labels, can be multiple labels per image separated by spaces
        with open(self.image_labels_file_path, "r") as f:
            # image path relative to the dataset path, label numbers, can be multiple separated by spaces
            image_labels = {
                line.split()[0]: [int(label) for label in line.split()[1:]]
                for line in f.readlines()
            }

        # expand sparse labels (e.g. 'image_file.jpg', [0,2]) to dense labels (e.g. 'image_file.jpg', [1,0,1])
        def expand_labels(image_labels: Dict[str, List[int]]) -> Dict[str, List[int]]:
            expanded_labels = {}
            for image_file, labels in image_labels.items():
                expanded_labels[image_file] = [
                    True if i in labels else False for i in range(len(label_names))
                ]
            return expanded_labels

        # filter out images without labels (from dense labels)
        def filter_labeled_images(
            image_labels: Dict[str, List[int]],
        ) -> Dict[str, List[int]]:
            return {
                image_file: labels
                for image_file, labels in image_labels.items()
                if len(labels) > 0
            }

        # store the expanded labels (e.g. 'image_file.jpg', [1,0,1])
        image_labels = expand_labels(filter_labeled_images(image_labels))
        # index: image file name, columns: label names
        self.image_labels_df = pd.DataFrame.from_dict(
            image_labels, orient="index", columns=label_names
        )

        # add metadata columns
        self.image_metadata_df = pd.DataFrame(
            index=self.image_labels_df.index, columns=["timestamp", "date"]
        )
        # add a column with the timestamp
        LENGTH_ISO8601_TIMESTAMP_WITH_TIMEZONE = 32
        # split of the file extension and get the timestamp part
        self.image_metadata_df["timestamp"] = self.image_metadata_df.index.str.split(
            "."
        ).str[0][-LENGTH_ISO8601_TIMESTAMP_WITH_TIMEZONE:]
        # convert the timestamp to a datetime object
        self.image_metadata_df["timestamp"] = pd.to_datetime(
            self.image_metadata_df["timestamp"], format="ISO8601"
        )

        # add a column with the date
        self.image_metadata_df["date"] = self.image_metadata_df["timestamp"].dt.date

    def __len__(self):
        return len(self.image_labels_df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get an image and its labels by index

        Parameters
        ----------
        idx : int
            The index of the image to get

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The image and its labels as numpy arrays (image dims: H x W x C, label dims: 1 x num_classes)
        """

        # load the image from disk
        image_file_name = self.image_labels_df.index[idx]
        image_file_path = self.root_dir / image_file_name
        image = io.imread(image_file_path)

        # apply the transformation
        if self.transform:
            image = self.transform(image)

        # get the labels as a numpy array, convert to float32 for PyTorch
        labels = self.image_labels_df.iloc[idx].values.astype(np.float32)

        return image, labels

    def get_integer_indices_for_labels(
        self, labels: List[str] | List[int]
    ) -> np.ndarray:
        """Get dataset indices of all images having at least all the given labels, possibly having additional labels

        Parameters
        ----------
        labels : List[str] | List[int]
            The selection of labels to get integer indices for, can be label names or label numbers

        Returns
        -------
        np.ndarray
            The list of integer indices having all at least the given labels, possibly having additional labels
        """
        if isinstance(labels[0], str):
            if not all([label in self.label_names for label in labels]):
                raise ValueError(
                    f"Invalid label name(s): {labels} (valid label names: {self.label_names})"
                )
            label_indices = [self.label_names.index(label) for label in labels]
        else:
            if not all([0 <= label < len(self.label_names) for label in labels]):
                raise ValueError(
                    f"Invalid label number(s): {labels} (valid label numbers: {list(range(len(self.label_names)))})"
                )
            label_indices = labels

        return np.where(self.image_labels_df.iloc[:, label_indices].all(axis=1))[0]

    def get_integer_indices_for_exclusive_labels(
        self, labels: List[str] | List[int]
    ) -> np.ndarray:
        """Get dataset indices of all images having exactly the given labels

        Parameters
        ----------
        labels : List[str] | List[int]
            The selection of labels to get integer indices for, can be label names or label numbers

        Returns
        -------
        np.ndarray
            The list of integer indices having exactly the given labels
        """
        if isinstance(labels[0], str):
            if not all([label in self.label_names for label in labels]):
                raise ValueError(
                    f"Invalid label name(s): {labels} (valid label names: {self.label_names})"
                )
            label_indices = [self.label_names.index(label) for label in labels]
        else:
            if not all([0 <= label < len(self.label_names) for label in labels]):
                raise ValueError(
                    f"Invalid label number(s): {labels} (valid label numbers: {list(range(len(self.label_names)))})"
                )
            label_indices = labels

        # create a list of boolean values, True for the selected labels, False for the rest
        expanded_label_indices = [
            True if i in label_indices else False for i in range(len(self.label_names))
        ]

        # for each line in the image_labels_df, compare if the labels are an exact match
        exact_match = np.logical_and.reduce(
            self.image_labels_df.values == np.array(expanded_label_indices), axis=1
        )

        # return the integer indices of the exact matches
        return np.where(exact_match)[0]
