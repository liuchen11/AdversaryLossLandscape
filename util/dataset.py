import torch
from torchvision import datasets, transforms

import numpy as np

def mnist(batch_size, data_augmentation = True, shuffle = True):

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST('./data', train = True, download = True, transform = transform)
    testset = datasets.MNIST('./data', train = False, download = True, transform = transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 1, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 1, pin_memory = True)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return train_loader, test_loader, classes

def cifar10(batch_size, data_augmentation = True, shuffle = True):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]) if data_augmentation == True else transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
    testset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle, num_workers = 4, pin_memory = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

