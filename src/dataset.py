import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Iterable, Literal
from pathlib import Path
from skimage import io

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SkyImageMultiLabelDataset(Dataset):
    '''Sky Image Multi-Label Dataset'''

    def __init__(self, root_dir: Path, image_labels_file: str = 'default.txt', label_names_file: str='synsets.txt', transform: transforms.Compose | None = None):
        '''Initialize the Sky Image Multi-Label Dataset

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
        '''
        
        self.root_dir = root_dir
        self.image_labels_file_path = root_dir / image_labels_file
        self.label_names_file_path = root_dir / label_names_file
        self.transform = transform
        
        # read the label names
        with open(self.label_names_file_path, 'r') as f:
            # line number corresponds to the label number, first line is label 0
            label_names = [ name.strip() for i, name in enumerate(f.readlines()) ]

        self.label_names = label_names
        
        # read the image labels, can be multiple labels per image separated by spaces
        with open(self.image_labels_file_path, 'r') as f:
            # image path relative to the dataset path, label numbers, can be multiple separated by spaces
            image_labels = { line.split()[0]: [int(label) for label in line.split()[1:]] for line in f.readlines() }
        
        # expand sparse labels (e.g. 'image_file.jpg', [0,2]) to dense labels (e.g. 'image_file.jpg', [1,0,1])
        def expand_labels(image_labels: Dict[str, List[int]]) -> Dict[str, List[int]]:
            expanded_labels = {}
            for image_file, labels in image_labels.items():
                expanded_labels[image_file] = [True if i in labels else False for i in range(len(label_names))]
            return expanded_labels

        # filter out images without labels (from dense labels)
        def filter_labeled_images(image_labels: Dict[str, List[int]]) -> Dict[str, List[int]]:
            return { image_file: labels for image_file, labels in image_labels.items() if len(labels) > 0 }

        # store the expanded labels (e.g. 'image_file.jpg', [1,0,1])
        image_labels = expand_labels(filter_labeled_images(image_labels))
        self.image_labels_df = pd.DataFrame.from_dict(image_labels, orient='index', columns=label_names)

    def __len__(self):
        return len(self.image_labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_file_name = self.image_labels_df.index[idx]
        image_file_path = self.root_dir / image_file_name
        image = io.imread(image_file_path)
        
        if self.transform:
            image = self.transform(image)
        
        sampel_labels_numerical = self.image_labels_df.iloc[idx].values
        sample_label_names = [ self.label_names[i] for i, label in enumerate(sampel_labels_numerical) if label ]
        sample = { 'image': image, 'labels': sampel_labels_numerical.astype(np.float32), 'filename': image_file_name, 'label_names': sample_label_names }

        return sample
