import numpy as np
import torch
import os
from PIL import Image

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = str(idx)+".png"
        image = os.path.join(self.image_dir, image )
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label