import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


class MultiLabelClassificationMobileNetV3Large(nn.Module):
    def __init__(self, num_classes, larger_input_size=False):
        """A multi-label classification model using a pretrained MobileNetV3-Large as the backbone and a custom classification head.

        Parameters
        ----------
        num_classes : int
            The number of classes the model should predict.
        larger_input_size : bool
            Whether to use a larger input size of 448x448 instead of the default 224x224.
        """
        super(MultiLabelClassificationMobileNetV3Large, self).__init__()
        # Take the pre-trained MobileNetV3 model, add a downsample layer to handle larger input images, and reinitialize the classification head
        
        # MobileNetV3-Large expects input images of size 224x224
        feature_extractor, avg_pool, _ = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ).children()
        
        if larger_input_size:
            
            # add a downsample layer to handle larger input images of size 448x448
            # the first element is the "features" module of the MobileNetV3-Large model
            # inside the "features" module, the first element is the first convolutional layer
            # we replace this first convolutional layer with our custom downsample layer
            first_conv_layer = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.Hardswish(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.Hardswish(),
            )
            
            # initialize the weights of the custom downsample layer
            for m in first_conv_layer.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
        else:
            # otherwise, use the default first convolutional layer
            first_conv_layer = feature_extractor[0]
        
        # build up the correct structure of mobile net v3 large again      
        self.feature_extractor = nn.Sequential(first_conv_layer, feature_extractor[1:])
        self.avg_pool = avg_pool
        
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
        x = self.avg_pool(self.feature_extractor(x))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
