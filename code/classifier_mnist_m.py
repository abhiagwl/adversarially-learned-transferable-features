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

class MNIST_M(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        self.train = train
        self.transform = transform
        if train:
            self.image_dir = os.path.join(root, 'train')
            labels_file = os.path.join(root, "train/train_labels.txt")
        else:
            self.image_dir = os.path.join(root, 'test')
            labels_file = os.path.join(root, "test/test_labels.txt")

        self.labels = np.loadtxt(labels_file).astype(np.long)
#         with open(labels_file, "r") as fp:
#             content = fp.readlines()
#         self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = str(idx)+".png"
        image = os.path.join(self.image_dir, image )
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label

    
root_dir = "./"
data_dir = os.path.join(root_dir,"datasets/mnistm/")

target_train_batch_size = 10240
target_test_batch_size = 1024

composed_transform = transforms.Compose([transforms.ToTensor()])

target_train_dataset = MNIST_M(root=data_dir, train=True, transform=composed_transform)
target_test_dataset = MNIST_M(root=data_dir, train=False, transform=composed_transform)

print('Size of train dataset: %d' % len(target_train_dataset))
print('Size of test dataset: %d' % len(target_test_dataset))

target_train_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=target_train_batch_size, shuffle=True)
target_test_loader = torch.utils.data.DataLoader(dataset=target_test_dataset, batch_size=target_test_batch_size, shuffle=False)

source_train_batch_size = 128
source_test_batch_size = 512
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

source_train_set = torchvision.datasets.MNIST(root='./datasets', train=True,
                                        download=True, transform=transform)
source_train_loader = torch.utils.data.DataLoader(source_train_set, batch_size=source_train_batch_size,
                                          shuffle=True, num_workers=2)


source_test_set = torchvision.datasets.MNIST(root='./datasets', train=False,
                                        download=True, transform=transform)
source_test_loader = torch.utils.data.DataLoader(source_test_set, batch_size=source_test_batch_size,
                                          shuffle=False, num_workers=2)

class MnistClassifier(nn.Module):
    def __init__(self, source_channels, target_channels, num_classes, ngpu=1):
        super(MnistClassifier, self).__init__()
        self.ngpu = ngpu
        self.private_source = nn.Sequential(
            nn.Conv2d(source_channels, 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.private_target = nn.Sequential(
            nn.Conv2d(target_channels, 32,5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2)
        )
        self.shared_convs = nn.Sequential(
            nn.Conv2d(32, 48, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
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
            output = nn.parallel.data_parallel(private_net, inputs, range(self.ngpu))
            output = nn.parallel.data_parallel(self.shared_convs, output, range(self.ngpu))
            output = output.view(output.size(0), -1)
            output = nn.parallel.data_parallel(self.shared_fcs, output, range(self.ngpu))
        else: 
            output = private_net(inputs)
            output = self.shared_convs(output)
            output = output.view(output.size(0), -1)
            output = self.shared_fcs(output)
        return output


criterion = nn.CrossEntropyLoss()

def test_acc(net, test_loader, Target=True):
    correct = 0
    total = 0
    net.eval()
    for data in test_loader:
        images, labels = data
        if is_gpu:
            images,labels = images.cuda(),labels.cuda()
        if Target==True:
            outputs = net(Variable(images))
        else :
            outputs = net(Variable(images),"source")
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        (100.0 * correct) / total))
    net.train()

def train_mnist_only(net,train_loader,test_loader,optimizer,criterion,epochs, Target_check=True):
    cudnn.benchmark = True
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
#             net.train()
            # get the inputs
    
            inputs, labels = data
            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()


            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs,"source")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            p_p = 200
            if i % p_p == (p_p-1):    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / p_p))
                running_loss = 0.0
#                 break
                test_acc(net,test_loader, Target_check)

    print('Finished Training')

def train_mnistm_only(net,train_loader,test_loader,optimizer,criterion,epochs,Target_check=True):
    for epoch in range(epochs):  # loop over the dataset multiple times
        net.train()
        cudnn.benchmark = True
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            if is_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs,"target")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            

            # print statistics
            running_loss += loss.data[0]
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
#                 break
        test_acc(net,test_loader,Target_check)
    print('Finished Training')

is_gpu = True
net = MnistClassifier(1, 3, 10,2)
if is_gpu:
    net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("Starting Training")
train_mnistm_only(net,target_train_loader, target_test_loader,optimizer,criterion,epochs=10)


# net = MnistClassifier(1, 3, 10)
# if is_gpu:
#     net.cuda()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# train_mnist_only(net,source_train_loader,source_test_loader,optimizer,criterion,epochs = 10,Target_check=False)
