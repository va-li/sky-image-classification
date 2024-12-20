import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


class MultiLabelClassificationMobileNetV3Large(nn.Module):
    def __init__(self, num_classes:int, image_input_size:int=224):
        """A multi-label classification model using a pretrained MobileNetV3-Large as the backbone and a custom classification head.

        Parameters
        ----------
        num_classes : int
            The number of classes the model should predict.
        image_input_size : int, either 224 or 448, optional, default=224 
            The input size of the image. If 448, the model will have a custom downsample layer to handle larger input images.
        """
        super(MultiLabelClassificationMobileNetV3Large, self).__init__()
        # Take the pre-trained MobileNetV3 model, add a downsample layer to handle larger input images, and reinitialize the classification head
        
        # MobileNetV3-Large expects input images of size 224x224
        feature_extractor, avg_pool, _ = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ).children()
        
        self.num_classes = num_classes
        self.image_input_size = image_input_size
        
        if image_input_size == 448:
            
            # add a downsample layer to handle larger input images of size 448x448
            # the first element is the "features" module of the MobileNetV3-Large model
            # inside the "features" module, the first element is the first convolutional layer
            # we replace this first convolutional layer with our custom downsample layer
            self.first_conv_layer = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(3),
                nn.Hardswish(),
            )
            
            # initialize the weights of the custom downsample layer
            for m in self.first_conv_layer.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                    
        elif image_input_size == 224:
            # otherwise, use the existing first convolutional layer
            self.first_conv_layer = feature_extractor[0]
            feature_extractor = feature_extractor[1:]
            
        else:
            raise ValueError(f"Only image_input_size of 224 or 448 is supported, but got {image_input_size}")
        
        # build up the correct structure of mobile net v3 large again      
        self.feature_extractor = nn.Sequential(self.first_conv_layer, feature_extractor)
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

    def freeze_pretrained_layers(self):
        """Freeze the pretrained layers of the model.
        
        If the model's `image_input_size` is 448, the first custom convolutional layer will be trainable,
        but the rest of the pretrained layers will be frozen.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # if the input size is 448, the first_conv_layer should be trainable
        # since it is a custom layer which was not part of the original MobileNetV3-Large model
        if self.image_input_size == 448:
            self.first_conv_layer.requires_grad = True
            
    def unfreeze_pretrained_layers(self):
        """Unfreeze the pretrained layers of the model."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
