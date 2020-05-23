import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip

from utils import plot_images

"""
loader class
"""

normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # cifar10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def _transform():
    valid_transform = Compose([
        ToTensor(),
        normalize
    ])

    train_transform = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
    ])

    return train_transform, valid_transform


def train_loaders(batch_size,
                  random_seed,
                  valid_size=0.2,
                  shuffle=True,
                  show_sample=False):
    train_transform, valid_transform = _transform()

    # load the dataset
    train_dataset = CIFAR10(root='..\\data', train=True, download=True, transform=train_transform)

    valid_dataset = CIFAR10(root='..\\data', train=True, download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(train_dataset, batch_size=9, shuffle=shuffle)
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        plot_images(X, [classes[label] for label in labels])

    return train_loader, valid_loader


def test_loader(batch_size,
                shuffle=True):
    # define transform
    transform = Compose([
        ToTensor(),
        normalize
    ])

    dataset = CIFAR10(root='..\\data', train=False, download=True, transform=transform)

    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)

    return test_loader
