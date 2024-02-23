# Torchvision datasets do not have a consistent API. Wrapper to make it so.
# Furthermore, utility to add a supervised flag to each observation.

import torchvision
import numpy as np

class CIFAR10:
    num_classes = 10
    imgsize = 32
    task = 'classification'

    def __init__(self, root, train, transform):
        self.ds = torchvision.datasets.CIFAR10(root, train)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        image, label = self.ds[i]
        image = np.array(image)
        return self.transform(image=image, label=label)

class VOCSEG:
    num_classes = 10
    task = 'segmentation'
    imgsize = 256

    def __init__(self, root, train, transform):
        image_set = 'train' if train else 'test'
        self.ds = torchvision.datasets.VOCSegmentation(root, image_set=image_set)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        print('d:', type(d))
        d = self.transform(**d)
        xx

class VOCDET:
    num_classes = 10  # background is class 0
    task = 'detection'
    imgsize = 256

    def __init__(self, root, train, transform):
        image_set = 'train' if train else 'test'
        self.ds = torchvision.datasets.VOCDetection(root, image_set=image_set)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        print('d:', type(d))
        d = self.transform(**d)
        xx
