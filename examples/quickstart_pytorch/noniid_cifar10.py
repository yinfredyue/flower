"""
Generate non-iid CIFAR-10 data for federated learning. 

Reference: https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-2-6c2e9494398b

Main changes made: 
- Non-iid testing dataloaders.
"""

import os
import random
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

# Ignore certain warning, for clearer output
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Use the same seed, for debugging
random.seed(0)

# Characteristics of Non-IID data
classes_pc = 2
num_clients = 5
batch_size = 32


def get_cifar10():
    """Return CIFAR10 train/test data and labels as numpy arrays"""
    data_train = torchvision.datasets.CIFAR10('.', train=True, download=True)
    data_test = torchvision.datasets.CIFAR10('.', train=False, download=True)

    x_train, y_train = data_train.data.transpose(
        (0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))


def clients_rand(train_len, nclients):
    """
    train_len: size of the train data
    nclients: number of clients

    Returns: a list of # of images each client should have.
    """
    client_tmp = []
    sum_ = 0
    #### creating random values for each client ####
    for i in range(nclients - 1):
        tmp = random.randint(10, 100)
        sum_ += tmp
        client_tmp.append(tmp)

    client_tmp = np.array(client_tmp)
    #### using those random values as weights ####
    clients_dist = ((client_tmp / sum_) * train_len).astype(int)
    num = train_len - clients_dist.sum()
    to_ret = list(clients_dist)
    to_ret.append(num)
    print("clients_rand ->", to_ret)
    return to_ret


def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    """
    Splits (data, labels) among n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels
      n_clients : number of clients
      classes_per_client : number of classes per client
      shuffle : True/False => True for shuffling the dataset, False otherwise
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : client data into desired format
    """
    #### constants ####
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    ### client distribution ####
    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(
        1, nd // classes_per_client) for nd in data_per_client]

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) ==
                           np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

        if verbose:
            print_split(clients_split)

    clients_split = np.array(clients_split)

    print("split_image_data -> list of length", len(clients_split),
          [(len(x), len(y)) for (x, y) in clients_split])
    return clients_split


def shuffle_list(data):
    """
    This function returns the shuffled data
    """

    def shuffle_list_data(x, y):
        """
        This function is a helper function, shuffles an
        array while maintaining the mapping between x and y
        """
        inds = list(range(len(x)))
        random.shuffle(inds)
        return x[inds], y[inds]

    for i in range(len(data)):
        tmp_len = len(data[i][0])
        index = [i for i in range(tmp_len)]
        random.shuffle(index)
        data[i][0], data[i][1] = shuffle_list_data(data[i][0], data[i][1])
    return data


class CustomImageDataset(Dataset):
    """
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    """

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(train=True, verbose=True):
    transforms_train = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    }
    transforms_eval = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train['cifar10'].transforms:
            print(' -', transformation)
        print()

    return (transforms_train['cifar10'], transforms_eval['cifar10'])


def get_data_loaders(nclients, batch_size, classes_pc=10, verbose=True):
    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    # data preprocessing & normalization
    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

    # Split training data
    train_split = split_image_data(x_train, y_train, n_clients=nclients,
                             classes_per_client=classes_pc, verbose=verbose)

    # Split testing data
    test_split = split_image_data(x_test, y_test, n_clients=nclients,
                             classes_per_client=classes_pc, verbose=verbose)

    # shuffle split train data
    train_split_shuffled = shuffle_list(train_split)

    # shuffle split test data
    test_split_shuffled = shuffle_list(test_split)

    # create training dataloaders
    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=batch_size, shuffle=True) for x, y in train_split_shuffled]

    # create testing dataloaders
    test_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_eval),
                                                  batch_size=batch_size, shuffle=True) for x, y in test_split_shuffled]

    return client_loaders, test_loaders


if __name__ == "__main__":
    print("get_data_loaders ->", get_data_loaders(num_clients, batch_size, classes_pc, True))
