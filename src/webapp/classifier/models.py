import numpy as np
import torch
import torch.nn as nn
from torchvision import models

class MultiLabelClassificationMobileNetV3Large(nn.Module):
    def __init__(self, num_classes):
        '''A multi-label classification model using a pretrained MobileNetV3-Large as the backbone and a custom classification head.

        Parameters
        ----------
        num_classes : int
            The number of classes the model should predict.
        '''
        super(MultiLabelClassificationMobileNetV3Large, self).__init__()
        # Take the pre-trained MobileNetV3 model and reinitialize the classification head
        self.backbone = nn.Sequential(*list(models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights).children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class MultiLabelClassificationMobileNetV3Small(nn.Module):
    def __init__(self, num_classes: int):
        '''A multi-label classification model using a pretrained MobileNetV3-Small as the backbone and a custom classification head.

        Parameters
        ----------
        num_classes : int
            The number of classes the model should predict.
        '''
        super(MultiLabelClassificationMobileNetV3Small, self).__init__()
        # Take the pre-trained MobileNetV3 model and reinitialize the classification head
        self.backbone = nn.Sequential(*list(models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights).children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
