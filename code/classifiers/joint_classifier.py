import numpy as np
import torch
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F

class MnistClassifier(nn.Module):
    def __init__(self, source_channels, target_channels, num_classes, rgpu=None,ngpu=1):
        super(MnistClassifier, self).__init__()
        self.ngpu = ngpu
        self.rgpu = rgpu
        if self.rgpu is None:
            self.rgpu = range(self.ngpu) 
        self.private_source = nn.Sequential(
            nn.Conv2d(source_channels, 32, 5),
            nn.MaxPool2d(2,2)
            nn.ReLU(True),
        )
        self.private_target = nn.Sequential(
            nn.Conv2d(target_channels, 32,5),
            nn.MaxPool2d(2,2)
            nn.ReLU(True),
        )
        self.shared_convs = nn.Sequential(
            nn.Conv2d(32, 48, 5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
        )
        self.shared_fcs = nn.Sequential(
            nn.Linear(16*48, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, num_classes)
        )


    def forward(self, inputs, dataset="target"):
        if dataset == "target":
            private_net = self.private_target
        else:
            private_net = self.private_source
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(private_net, inputs, self.rgpu)
            output = nn.parallel.data_parallel(self.shared_convs, output, self.rgpu)
            output = output.view(output.size(0), -1)
            output = nn.parallel.data_parallel(self.shared_fcs, output, self.rgpu)
        else :
            output = private_net(inputs)
            output = self.shared_convs(output)
            output = output.view(output.size(0), -1)
            output = self.shared_fcs(output)
        return output