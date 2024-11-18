#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SkyImageMultiLabelDataset
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from pathlib import Path
from tqdm import tqdm
import time
import json
import logging
import matplotlib.pyplot as plt

# each training run gets its own directory to store the model, metrics and logs
training_run_timestamp = time.strftime('%Y%m%d-%H%M%S%z')
training_run_data_path = Path(f'/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/training-runs/mobilenetv3_{training_run_timestamp}')
training_run_data_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    # log to stdout and to a file
    handlers=[logging.StreamHandler(), logging.FileHandler(training_run_data_path / 'training.log')]
)

# Define transformations for the training, validation and test sets
transform = transforms.Compose([
    transforms.ToTensor(), # first, convert image (numpy array) to PyTorch tensor, so that further processing can be done
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hyperparameters dictionary
hyperparameters = {}

#%%
# Load the dataset
dataset_path = Path('/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/')
hyperparameters['dataset_path'] = str(dataset_path)
dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform)
hyperparameters['transform'] = str(transform)

# set the random seed for reproducibility
TORCH_SEED = 18759
torch.manual_seed(TORCH_SEED)
hyperparameters['torch_seed'] = TORCH_SEED

# Split the dataset into training, validation and test sets
TRAIN_SIZE_RATIO = 0.8
train_size = int(TRAIN_SIZE_RATIO * len(dataset))
hyperparameters['train_size_ratio'] = TRAIN_SIZE_RATIO
VAL_SIZE_RATIO = 0.1
val_size = int(VAL_SIZE_RATIO * len(dataset))
hyperparameters['val_size_ratio'] = VAL_SIZE_RATIO
TEST_SIZE_RATIO = 0.1
test_size = len(dataset) - train_size - val_size
hyperparameters['test_size_ratio'] = TEST_SIZE_RATIO

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_test_val_indices = {
    'train': train_dataset.indices,
    'val': val_dataset.indices,
    'test': test_dataset.indices
}
hyperparameters['dataset_indices'] = train_test_val_indices

BATCH_SIZE = 32
hyperparameters['batch_size'] = BATCH_SIZE

SUFFLE_TRAIN = True
hyperparameters['shuffle_train'] = SUFFLE_TRAIN
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SUFFLE_TRAIN)
SUFFLE_VAL = False
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SUFFLE_VAL)
hyperparameters['shuffle_val'] = SUFFLE_VAL
SHUFFLE_TEST = False
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE_TEST)
hyperparameters['shuffle_test'] = SHUFFLE_TEST

#%%
logging.info(f'Train set:\n{dataset.image_labels_df.iloc[train_dataset.indices].sum()}')
logging.info(f'Validation set:\n{dataset.image_labels_df.iloc[val_dataset.indices].sum()}')
logging.info(f'Test set:\n{dataset.image_labels_df.iloc[test_dataset.indices].sum()}')
#%%
# Load the pre-trained MobileNetV3 model
# model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights)
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights)
hyperparameters['model'] = 'mobilenet_v3_large'

# Replace the classification head
# mobileNetV3_small
# model.classifier = nn.Sequential(
#   nn.Linear(in_features=576, out_features=1024, bias=True),
#   nn.Hardswish(),
#   nn.Dropout(p=0.2, inplace=True),
#   nn.Linear(in_features=1024, out_features=len(dataset.label_names), bias=True),
#   nn.Sigmoid()
# )

# mobileNetV3_large
model.classifier = nn.Sequential(
  nn.Linear(in_features=960, out_features=1280, bias=True),
  nn.Hardswish(),
  nn.Dropout(p=0.2, inplace=True),
  nn.Linear(in_features=1280, out_features=len(dataset.label_names), bias=True),
  nn.Sigmoid()
)

hyperparameters['classifier'] = str(model.classifier)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
# BCELoss is used for multi-label classification
criterion = nn.BCELoss()
LEARNING_RATE = 0.001
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
hyperparameters['learning_rate'] = LEARNING_RATE

# # learning rate scheduler
# LR_SCHEDULE = 'StepLR'
# LR_SCHEDULE_STEP_SIZE = 5
# LR_SCHEDULE_GAMMA = 0.1
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULE_STEP_SIZE, gamma=LR_SCHEDULE_GAMMA)
# hyperparameters['lr_schedule'] = LR_SCHEDULE
# hyperparameters['lr_schedule_step_size'] = LR_SCHEDULE_STEP_SIZE
# hyperparameters['lr_schedule_gamma'] = LR_SCHEDULE_GAMMA

# Training loop
N_EPOCHS = 20
hyperparameters['n_epochs'] = N_EPOCHS
FRZAE_LAYERS = False
hyperparameters['freeze_layers'] = FRZAE_LAYERS
best_val_loss = float('inf')
best_val_loss_epoch = 0
train_losses = []
train_times = []
val_losses = []
val_accuracies = []
val_times = []

(training_run_data_path / 'hyperparameters.json').write_text(json.dumps(hyperparameters, indent=4))

for epoch in range(N_EPOCHS):
    
    train_start = time.time()
        
    model.train()
    # freeze all layers except the classifier
    if FRZAE_LAYERS:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{N_EPOCHS}'):
        inputs, labels = inputs.to(device), labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_duration = time.time() - train_start
    train_times.append(train_duration)
    
    val_start = time.time()
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{N_EPOCHS}'):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            preds = outputs > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_duration = time.time() - val_start
    val_times.append(val_duration)
    
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_accuracy)
    
    logging.info(f'Results Epoch {epoch+1}/{N_EPOCHS}')
    
    # show confusion matrix with label names
    # as a nice table labeled with TRUE/FALSE for ground truth and predicted
    conf_matrix = multilabel_confusion_matrix(all_labels, all_preds)
    for i, conf in enumerate(conf_matrix):
        tp = conf[1,1]
        tn = conf[0,0]
        fp = conf[0,1]
        fn = conf[1,0]
        
        logging.info(f'Label: {dataset.label_names[i]}')
        # show it as a table
        logging.info('\n' + str(pd.DataFrame(conf, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1'])))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        
        logging.info(f'Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')
        
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_loss_epoch = epoch
        torch.save(model.state_dict(), training_run_data_path /  'best_model.pth')
        logging.info(f'New best model saved at epoch {epoch+1}')
        
    # save the metrics
    metrics = pd.DataFrame({
        'train_loss': train_losses,
        'train_time': train_times,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_time': val_times
    }, index=range(1, epoch+2))
    metrics.to_csv(training_run_data_path / 'train_metrics.csv', index_label='epoch')
    
    logging.info(f'Epoch {epoch+1} summary:')
    logging.info(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Time: train {train_duration:.0f}s + val {val_duration:.0f}s')

    # create plots of the training metrics, loss and accuracy in the same plot (left axis for loss, right axis for accuracy)
    metrics = pd.read_csv(training_run_data_path / 'train_metrics.csv', index_col='epoch')
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(metrics['train_loss'], color=color, label='Train Loss')
    ax1.plot(metrics['val_loss'], color=color, linestyle='dashed', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    # legend in upper left corner
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(metrics['val_accuracy'], color=color, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    # legend in upper right corner
    ax2.legend(loc='upper right')
    
    # set the x-axis to be the epoch number
    ax1.set_xticks(metrics.index)
    ax1.set_xticklabels(metrics.index)
    
    fig.tight_layout()    
    # save the plot
    plt.savefig(training_run_data_path / 'metrics_plot.png')

logging.info('Training complete.')
