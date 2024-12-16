import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


class MultiLabelClassificationMobileNetV3Large(nn.Module):
    def __init__(self, num_classes):
        """A multi-label classification model using a pretrained MobileNetV3-Large as the backbone and a custom classification head.

        Parameters
        ----------
        num_classes : int
            The number of classes the model should predict.
        """
        super(MultiLabelClassificationMobileNetV3Large, self).__init__()
        # Take the pre-trained MobileNetV3 model, add a downsample layer to handle larger input images, and reinitialize the classification head
        
        # MobileNetV3-Large expects input images of size 224x224
        self.backbone = nn.Sequential(
            *list(
                models.mobilenet_v3_large(
                    weights=models.MobileNet_V3_Large_Weights.DEFAULT
                ).children()
            )[:-1] # remove the classification head
        )
        
        custom_downsampler = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.Hardswish(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )
        
        # initialize the weights of the custom downsample layer
        for m in custom_downsampler.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
        
        # add a downsample layer to handle larger input images of size 448x448
        # the first element is the "features" module of the MobileNetV3-Large model
        # inside the "features" module, the first element is the first convolutional layer
        # we replace this first convolutional layer with our custom downsample layer
        self.backbone[0][0] = custom_downsampler
        
        # then we add a custom classification head
        # which we use separately in the forward pass
        self.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True),
            nn.Sigmoid(),
        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

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
