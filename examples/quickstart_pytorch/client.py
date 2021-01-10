import sys

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from noniid_cifar10 import get_data_loaders

import flwr as fl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
    class Net(nn.Module):
        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(), len(trainloader)

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return len(testloader), float(loss), float(accuracy)

    # Start client, parsing arguments
    argv = sys.argv[1:]

    # num of clients
    if len(argv) < 2:
        print("Usage: python client.py idx num_clients [min_delay max_delay]")
        exit(1)

    idx = int(argv[0])
    num_clients = int(argv[1])
    if idx >= num_clients:
        print("Usage: idx should be zero-indexed")
        exit(1)

    print(f"I'm client {idx}")

    # delay related
    delay = False
    min_delay, max_delay = 0, 0
    try:
        if len(argv) == 3:
            min_delay = int(argv[2])
            max_delay = min_delay
            delay = True
        elif len(argv) == 4:
            delay = True
            min_delay = int(argv[2])
            max_delay = int(argv[3])
            delay = True
    except:
        pass
    if delay:
        print(f"Start SSP client with delay [{min_delay},{max_delay}]")
    else:
        print("Start SSP client without delay")

    # Load data (CIFAR-10)
    trainloader, testloader = load_data(idx, num_clients)

    fl.client.start_numpy_client_ssp("[::]:8080", client=CifarClient(
    ), delay=delay, min_delay=min_delay, max_delay=max_delay)


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def load_data(idx, num_clients, noniid=True):
    """Load CIFAR-10 (training and test set)."""
    if noniid:
        # Non-iid
        train_loaders, test_loaders = get_data_loaders(num_clients, 32, 5, False)
        return train_loaders[idx], test_loaders[idx]
    else:
        # iid data, all clients use the same dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = CIFAR10(".", train=True, download=True, transform=transform)
        testset = CIFAR10(".", train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32)
        return trainloader, testloader

if __name__ == "__main__":
    main()
