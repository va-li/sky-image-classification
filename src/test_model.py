#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
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

from dataset import SkyImageMultiLabelDataset
from models import MultiLabelClassificationMobileNetV3Large

training_run_data_path = Path(f'/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/training-runs/mobilenetv3_20241206-115523+0100')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # log to stdout and to a file
    handlers=[logging.StreamHandler(), logging.FileHandler(training_run_data_path / 'test.log')]
)

# 1. load data
# 2. load model
# 3. evaluate model on test data
# 4. save evaluation results

# 1. load data
#%%
try:
    
    # load the test dataset
    dataset_path = Path('/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/')
    hyperparameters_train = json.loads((training_run_data_path / 'hyperparameters.json').read_text())
    
    transform_test_test = v2.Compose([
        v2.ToTensor(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_test_test)
    test_indices = hyperparameters_train['dataset_indices']['test']
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    SHUFFLE_TEST = False
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters_train['batch_size'], shuffle=False)
    
    device = 'cpu'
    
    logging.info(f'Test set:\n{dataset.image_labels_df.iloc[test_dataset.indices].sum()}')
    
    # load the model
    model = MultiLabelClassificationMobileNetV3Large(num_classes=len(dataset.label_names))
    try:
        model.load_state_dict(torch.load(training_run_data_path / 'best_model.pth', map_location=torch.device(device)))
    except FileNotFoundError:
        print("No pretrained weights found. Model will use random initialization.")

    model.eval()
    
    # evaluate model on test data
    
    test_start = time.time()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            
            preds = outputs > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_duration = time.time() - test_start
    
    # show confusion matrix with label names and accuracy, recall, precision for each class in the validation set
    # as a nice table labeled with TRUE/FALSE for ground truth and predicted
    conf_matrix = multilabel_confusion_matrix(all_labels, all_preds)
    class_accuracies = []
    class_recalls = []
    class_precisions = []
    for i, conf in enumerate(conf_matrix):
        tp = conf[1,1]
        tn = conf[0,0]
        fp = conf[0,1]
        fn = conf[1,0]
        
        # show it as a table
        logging.info(f'Label: {dataset.label_names[i]}\n' + str(pd.DataFrame(conf, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1'])))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        class_accuracies.append(accuracy)
        recall = tp / (tp + fn)
        class_recalls.append(recall)
        precision = tp / (tp + fp)
        class_precisions.append(precision)
        
        logging.info(f'Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')
    
    # calculate the subset accuracy and jaccard score for the validation set
    test_subset_accuracy = accuracy_score(all_labels, all_preds)
    test_mean_jaccard = jaccard_score(all_labels, all_preds, average='samples') # calculated for each sample and then averaged
    test_mean_precision = np.nanmean(class_precisions)
    test_mean_recall = np.mean(class_recalls)
    test_mean_accuracy = np.mean(class_accuracies)
    
    logging.info(f'Test time {test_duration:.0f}s')
    logging.info(f'Test Mean Accuracy: {test_mean_accuracy:.4f}, Test Mean Precision: {test_mean_precision:.4f}, Test Mean Recall: {test_mean_recall:.4f}')
    logging.info(f'Test Subset Accuracy: {test_subset_accuracy:.4f}, Test Mean Jaccard: {test_mean_jaccard:.4f}')
    
    logging.info('Finished evaluation')
    
except Exception as e:
    # print the full traceback to the log file
    logging.exception('Exception occurred')
    
    # re-raise the exception to see the full traceback in the notebook
    raise e
