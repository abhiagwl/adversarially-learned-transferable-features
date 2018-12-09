import numpy as np
import torch
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, source_channels, target_channels, z, rgpu=None,ngpu=1):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.rgpu = rgpu
        self.z = z
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
            nn.ReLU(True)
        )
        self.mean = nn.Sequential(
            nn.Linear(100, 100),
        )
        self.logvar = nn.Sequential(
            nn.Linear(100, 100)
        )
    def sample_z(mean,logvar,batch_size):
        std = logvar.mul(0.5).exp_()        
        noise = Variable(torch.randn((batch_size, self.z))).cuda()
        sampled_z = mean + std * noise
        return repa
    
    def forward(self, inputs, dataset="target"):
        if dataset == "target":
            private_net = self.private_target
        else:
            private_net = self.private_source
        batch_size = inputs.size()[0]
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(private_net, inputs, self.rgpu)
            output = nn.parallel.data_parallel(self.shared_convs, output, self.rgpu)
            output = output.view(output.size(0), -1)
            output = nn.parallel.data_parallel(self.shared_fcs, output, self.rgpu)
            mean = nn.parallel.data_parallel(self.mean, output, self.rgpu)
            logvar = nn.parallel.data_parallel(self.logvar, output, self.rgpu)
            output = sampled_z(mean,logvar,batch_size)
            
        else :
            output = private_net(inputs)
            output = self.shared_convs(output)
            output = output.view(output.size(0), -1)
            output = self.shared_fcs(output)
        return (mean,logvar,output)
class TaskNet(nn.Module):
    def __init__(self, num_classes,z, rgpu=None,ngpu=1):
        super(TaskNet, self).__init__()
        self.ngpu = ngpu
        self.rgpu = rgpu
        self.z =z
        if self.rgpu is None:
            self.rgpu = range(self.ngpu) 
        self.shared_fcs = nn.Sequential(
            # nn.Linear(100, 100),
            # nn.ReLU(True),
            nn.Linear(self.z, num_classes)
        )
    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.shared_fcs, inputs, self.rgpu)
        else :
            output = self.shared_fcs(inputs)
        return output

class Generator(nn.Module):
    # initializers
    def __init__(self, d=128,z=100,c=3,ngpu=1,rgpu=None):
        super(generator, self).__init__()
        self.ngpu=ngpu
        self.rgpu = rgpu
        self.z = z
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(z, d * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(d * 8),
                nn.ReLU(True),
                # state size. (d*8) x 4 x 4
                nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 4),
                nn.ReLU(True),
                # state size. (d*4) x 8 x 8
                nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d * 2),
                nn.ReLU(True),
                # state size. (d*2) x 16 x 16
                nn.ConvTranspose2d(d * 2,     d, 4, 2, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                # state size. (d) x 32 x 32
                nn.ConvTranspose2d(    d,      c, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
                )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.rgpu)
        else:
            output = self.main(input)
        return output
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128,c=3,ngpu=1,rgpu = None):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
        self.rgpu = rgpu
        self.main = nn.Sequential(
            # input is (c) x 64 x 64
            nn.Conv2d(c, d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d) x 32 x 32
            nn.Conv2d(d, d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d*2) x 16 x 16
            nn.Conv2d(d * 2, d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d*4) x 8 x 8
            nn.Conv2d(d * 4, d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (d*8) x 4 x 4
            nn.Conv2d(d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(s elf, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, self.rgpu)
        else:
            output = self.main(input)

        return output#.view(-1, 1).squeeze(1)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
