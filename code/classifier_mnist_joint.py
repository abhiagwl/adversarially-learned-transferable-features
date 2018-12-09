import pickle as pkl
import numpy as np
import tensorflow as tf
import torch
import os
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import torchvision.transforms as transforms
import torch.nn as nn
from torch.backends import cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

from classifiers.joint_classifier import MnistClassifier
from dataloaders.concat_dataset import ConcatDataset
from dataloaders.mnistm_loader import MNIST_M
from utils.train_utils import train_mnist_joint

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, required=True)
parser.add_argument('-rg', '--range_of_the_gpu', nargs='+', type=int, required=False)
parser.add_argument('-e', '--num_of_epoch', type=int, required=True)
parser.add_argument('-tbs', '--target_batch_size', type=int, required=False)
parser.add_argument('-sbs', '--source_batch_size', type=int, required=False)
parser.add_argument('-lr', '--learn_rate', type=float, required=False)
parser.add_argument('-mew', '--momentum', type=float, required=False)
parser.add_argument('-slr','--lear_rate_schedule', nargs='+', type=float, required=False)
parser.add_argument('-estp', '--early_stop', type=int, required=False)
args = parser.parse_args()
ngpu = args.gpu
epochs = args.num_of_epoch

if args.early_stop is not None:
    estp = args.early_stop
else:
    estp = None
if args.lear_rate_schedule is not None:
    slr = args.lear_rate_schedule
else:
    slr = [0.5,0.75]

if args.range_of_the_gpu is not None:
    rgpu = args.range_of_the_gpu
else:
    rgpu = range(ngpu)
if args.target_batch_size is not None:
    target_train_batch_size = args.target_batch_size
else :
    target_train_batch_size = 32

if args.source_batch_size is not None:
    source_train_batch_size = args.source_batch_size
else :
    source_train_batch_size = 32
if args.learn_rate is not None:
    lr = args.learn_rate
else :
    lr = 0.001
if args.momentum is not None:
    mew = args.momentum
else :
    mew = 0.9
    
    
target_test_batch_size = 512
source_test_batch_size = 512

root_dir = "../"
data_dir = os.path.join(root_dir,"datasets/")
mnistm_dir = os.path.join(root_dir,"datasets/mnistm/")


target_transform = transforms.Compose([transforms.ToTensor()])
source_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

target_train_dataset = MNIST_M(root=mnistm_dir, train=True, transform=target_transform)
target_test_dataset = MNIST_M(root=mnistm_dir, train=False, transform=target_transform)

print('Size of target train dataset: %d' % len(target_train_dataset))
print('Size of target test dataset: %d' % len(target_test_dataset))


source_train_set = torchvision.datasets.MNIST(root=data_dir, train=True,
                                        download=True, transform=source_transform)
source_test_set = torchvision.datasets.MNIST(root=data_dir, train=False,
                                        download=True, transform=source_transform)

print('Size of source train dataset: %d' % len(target_train_dataset))
print('Size of source test dataset: %d' % len(target_test_dataset))

train_loader = torch.utils.data.DataLoader(
            ConcatDataset(source_train_set,target_train_dataset),
            batch_size=source_train_batch_size, shuffle=True, pin_memory=False)
test_loader = torch.utils.data.DataLoader(
            ConcatDataset(source_test_set,target_test_dataset),
            batch_size=source_test_batch_size, shuffle=True, pin_memory=False)

net = MnistClassifier(1, 3, 10,rgpu,ngpu)
# if is_gpu:
net.cuda(rgpu[0])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mew)

train_mnist_joint(net,train_loader,train_loader,optimizer,criterion,epochs = epochs,early_stop = estp,schedule=slr)