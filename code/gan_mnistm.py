import os, time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

import torchvision.utils as utils

import numpy as np

is_gpu = torch.cuda.is_available()

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128,z=100,c=3,ngpu=1):
        super(generator, self).__init__()
        self.ngpu=ngpu
        self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     z, d * 8, 4, 1, 0, bias=False),
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
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128,c=3,ngpu=1):
        super(discriminator, self).__init__()
        self.ngpu = ngpu
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


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
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

fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

batch_size = 128
lr = 0.0002
train_epoch = 20

# data_loader
img_size = 64

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

train_batch_size = 128
test_batch_size = 128
# source_train_batch_size = 32
# source_test_batch_size = 512

# composed_transform = transforms.Compose([transforms.ToTensor()])
composed_transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_dataset = MNIST_M(root=data_dir, train=True, transform=composed_transform)
# test_dataset = MNIST_M(root=data_dir, train=False, transform=composed_transform)

print('Size of train dataset: %d' % len(train_dataset))
# print('Size of test dataset: %d' % len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./datasets/', train=True, download=True, transform=transform),
#     batch_size=batch_size, shuffle=True)

# def imshow(img):
#     npimg = img.numpy()
#     npimg = npimg*0.5 + 0.5
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# train_dataiter = iter(target_train_loader)
# train_images, train_labels = train_dataiter.next()
# # print("Train images", train_images)
# # print("Train images", train_labels)
# imshow(utils.make_grid(train_images[:2]))

# network
G = generator(128,ngpu=4)
D = discriminator(128,ngpu=4)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
if is_gpu:
    G.cuda()
    D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('MNISTM_DCGAN_results'):
    os.mkdir('MNISTM_DCGAN_results')
if not os.path.isdir('MNISTM_DCGAN_results/Random_results'):
    os.mkdir('MNISTM_DCGAN_results/Random_results')
if not os.path.isdir('MNISTM_DCGAN_results/Fixed_results'):
    os.mkdir('MNISTM_DCGAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
num_iter = 0

print('training start!')

start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    
    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    num_iter = 0

    epoch_start_time = time.time()
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)
        if is_gpu:
            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        else:
            x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)

        D_result = D(x_).squeeze()
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        if is_gpu:
            z_ = Variable(z_.cuda())
        else:
            z_ = Variable(z_)
        G_result = G(z_)

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # D_losses.append(D_train_loss.data[0])
        D_losses.append(D_train_loss.data[0])

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)#heuristic loss by using y_real imposes maximization
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data[0])

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    p = 'MNISTM_DCGAN_results/Random_results/MNISTM_DCGAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNISTM_DCGAN_results/Fixed_results/MNISTM_DCGAN_' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=p, isFix=False)
    show_result((epoch+1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)



print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "MNISTM_DCGAN_results/generator_param.pkl")
torch.save(D.state_dict(), "MNISTM_DCGAN_results/discriminator_param.pkl")
with open('MNISTM_DCGAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, show=True,save=True, path='MNISTM_DCGAN_results/MNISTM_DCGAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNISTM_DCGAN_results/Fixed_results/MNISTM_DCGAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNISTM_DCGAN_results/generation_animation.gif', images, fps=5)

# from IPython.display import Image as gif_view
# gif_view(url="MNISTM_DCGAN_results/generation_animation.gif"
