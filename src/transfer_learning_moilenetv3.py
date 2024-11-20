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

try:

    # Define transformations for the training, validation and test sets
    transform_train = v2.Compose([
        v2.ToTensor(), # first, convert image (numpy array) to PyTorch tensor, so that further processing can be done
        v2.RandomHorizontalFlip(), # randomly flip and rotate
        v2.RandomRotation((0, 180)), # randomly rotate the image between 0 and 180 degrees
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # randomly change the brightness, contrast, saturation and hue
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val_test = v2.Compose([
        v2.ToTensor(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Hyperparameters dictionary
    hyperparameters = {}

    #%%
    # Load the dataset
    dataset_path = Path('/home/vbauer/MEGA/Master/Data Science/2024 WS/Applied Deep Learning/sky-image-classification/data/')
    hyperparameters['dataset_path'] = str(dataset_path)
    dataset = SkyImageMultiLabelDataset(dataset_path)
    train_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_train)
    hyperparameters['transform_train'] = str(transform_train)
    val_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_val_test)
    test_dataset = SkyImageMultiLabelDataset(dataset_path, transform=transform_val_test)
    hyperparameters['transform_val_test'] = str(transform_val_test)

    # set the random seed for reproducibility
    SEED = 18759
    torch.manual_seed(SEED)
    hyperparameters['seed'] = SEED
    

    # Split the dataset into training, validation and test sets
    VAL_SIZE_RATIO = 0.1
    VAL_SIZE = int(VAL_SIZE_RATIO * len(dataset))
    hyperparameters['val_size_ratio'] = VAL_SIZE_RATIO
    TEST_SIZE_RATIO = 0.1
    TEST_SIZE = int(TEST_SIZE_RATIO * len(dataset))
    hyperparameters['test_size_ratio'] = TEST_SIZE_RATIO
    TRAIN_SIZE_RATIO = 1 - VAL_SIZE_RATIO - TEST_SIZE_RATIO
    TRAIN_SIZE = len(dataset) - VAL_SIZE - TEST_SIZE
    hyperparameters['train_size_ratio'] = TRAIN_SIZE_RATIO
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=TEST_SIZE, random_state=SEED)
    train_indices, val_indices = train_test_split(train_indices, test_size=VAL_SIZE, random_state=SEED)
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

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
    # Initialize the model
    model = MultiLabelClassificationMobileNetV3Large(num_classes=len(dataset.label_names))

    hyperparameters['model'] = 'MultiLabelClassificationMobileNetV3Large'
    hyperparameters['classifier'] = str(model.classifier)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    # BCELoss is used for multi-label classification
    bce_weights = 1 / dataset.image_labels_df.iloc[train_dataset.indices].mean()
    hyperparameters['loss_function_weights'] = bce_weights.to_dict()
    criterion = nn.BCELoss(weight=torch.tensor(bce_weights.values).float().to(device))
    LEARNING_RATE = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    hyperparameters['learning_rate'] = LEARNING_RATE

    # Training loop
    N_EPOCHS = 20
    hyperparameters['n_epochs'] = N_EPOCHS
    FRZAE_LAYERS = True
    hyperparameters['freeze_layers'] = FRZAE_LAYERS
    best_val_loss = float('inf')
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

    (training_run_data_path / 'hyperparameters.json').write_text(json.dumps(hyperparameters, indent=4))

    for epoch in range(N_EPOCHS):
        
        train_start = time.time()
            
        model.train()
        # freeze all layers except the classifier
        if FRZAE_LAYERS:
            model.freeze_backbone()
            
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
        
        logging.info(f'Results Epoch {epoch+1}/{N_EPOCHS}')
        
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
            
            logging.info(f'Label: {dataset.label_names[i]}')
            # show it as a table
            logging.info('\n' + str(pd.DataFrame(conf, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1'])))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            class_accuracies.append(accuracy)
            recall = tp / (tp + fn)
            class_recalls.append(recall)
            precision = tp / (tp + fp)
            class_precisions.append(precision)
            
            logging.info(f'Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}')
        
        # calculate the subset accuracy and jaccard score for the validation set
        val_subset_accuracy = accuracy_score(all_labels, all_preds)
        val_subset_accuracies.append(val_subset_accuracy)
        val_mean_jaccard = jaccard_score(all_labels, all_preds, average='samples') # calculated for each sample and then averaged
        val_mean_jaccards.append(val_mean_jaccard)
        # calculate the mean accuracy, precision and recall for the validation set
        val_mean_precisions.append(np.nanmean(class_precisions))
        val_mean_recalls.append(np.mean(class_recalls))
        val_mean_accuracies.append(np.mean(class_accuracies))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            torch.save(model.state_dict(), training_run_data_path /  'best_model.pth')
            logging.info(f'New best model saved at epoch {epoch+1}')
        
        logging.info(f'Epoch {epoch+1} summary:')
        logging.info(f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: train {train_duration:.0f}s + val {val_duration:.0f}s')
        logging.info(f'Val Mean Accuracy: {val_mean_accuracies[-1]:.4f}, Val Mean Precision: {val_mean_precisions[-1]:.4f}, Val Mean Recall: {val_mean_recalls[-1]:.4f}')
        logging.info(f'Val Subset Accuracy: {val_subset_accuracy:.4f}, Val Mean Jaccard: {val_mean_jaccard:.4f}')
            
        # save the metrics
        metrics = pd.DataFrame({
            'train_loss': train_losses,
            'train_time': train_times,
            'val_loss': val_losses,
            'val_subset_accuracy': val_subset_accuracies,
            'val_mean_accuracy': val_mean_accuracies,
            'val_mean_jaccard': val_mean_jaccards,
            'val_mean_precision': val_mean_precisions,
            'val_mean_recall': val_mean_recalls,
            'val_time': val_times
        }, index=range(1, epoch+2))
        metrics.to_csv(training_run_data_path / 'train_metrics.csv', index_label='epoch')

        # create plots of the training metrics, loss and accuracy in the same plot (left axis for loss, right axis for accuracy)
        metrics = pd.read_csv(training_run_data_path / 'train_metrics.csv', index_col='epoch')
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(metrics['train_loss'], color=color, label='Training')
        ax1.plot(metrics['val_loss'], color=color, linestyle='dashed', label='Validation')
        ax1.tick_params(axis='y', labelcolor=color)
        # legend in upper left corner
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Subset Accuracy', color=color)
        ax2.plot(metrics['val_subset_accuracy'], color=color, label='Validation')
        ax2.tick_params(axis='y', labelcolor=color)
        # legend in upper right corner
        ax2.legend(loc='upper right')
        
        # set the x-axis to be the epoch number
        ax1.set_xticks(metrics.index)
        ax1.set_xticklabels(metrics.index)
        
        fig.tight_layout()
        fig.suptitle(f'{training_run_data_path.name} - Training Metrics')
        # save the plot
        plt.savefig(training_run_data_path / 'metrics_plot.png')
        
        # plot mean jaccard, mean accuracy, mean precision and mean recall 
        fig, ax = plt.subplots()
        ax.plot(metrics['val_mean_jaccard'], label='Mean Jaccard')
        ax.plot(metrics['val_mean_accuracy'], label='Mean Accuracy')
        ax.plot(metrics['val_mean_precision'], label='Mean Precision')
        ax.plot(metrics['val_mean_recall'], label='Mean Recall')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.set_xticks(metrics.index)
        ax.set_xticklabels(metrics.index)
        fig.suptitle(f'{training_run_data_path.name} - Validation Metrics')
        fig.tight_layout()
        plt.savefig(training_run_data_path / 'validation_metrics_plot.png')

    logging.info('Training complete.')
    
except Exception as e:
    # print the full traceback to the log file
    logging.exception('Exception occurred')
    
    # re-raise the exception to see the full traceback in the notebook
    raise e
