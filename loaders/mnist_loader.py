import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip

from utils import plot_images

"""
loader class
"""

normalize = Normalize((0.1307,), (0.3081,))  # MNIST

def _transform(augment):

    valid_transform = Compose([
        ToTensor(),
        normalize
    ])

    if augment:
        train_transform = Compose([
            Resize(240),
            RandomCrop(224, padding=10),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize
        ])
    else:
        train_transform = Compose([
            ToTensor(),
            normalize
        ])

    return train_transform, valid_transform


def train_loaders(batch_size,
                  random_seed,
                  augment=False,
                  valid_size=0.2,
                  shuffle=True,
                  show_sample=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    """
    train_transform, valid_transform = _transform(augment)

    # load the dataset
    train_dataset = MNIST(root='..\\data', train=True, download=True, transform=train_transform)

    valid_dataset = MNIST(root='..\\data', train=True, download=True, transform=valid_transform)

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
        plot_images(X, [int(i) for i in labels])

    return train_loader, valid_loader


def test_loader(batch_size,
                shuffle=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    """

    # define transform
    transform = Compose([
        ToTensor(),
        normalize
    ])

    dataset = MNIST(root='..\\data', train=False, download=True, transform=transform)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader
