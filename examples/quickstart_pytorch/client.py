import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from models import *
from noniid_cifar10 import get_data_loaders, get_full_test_dataloader

import flwr as fl
from flwr.common.switchpoint import TestStrategy, AccuracyVariance
from flwr.common.logger import log
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = VGG11()
DATA_FRACTION = 0.1
BATCH_SIZE=32

def main():
    # python client.py --num_clients 2 --staleness_bound 2 --idx 0
    # python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.4:8080 --idx 0
    """Create model, load data, define Flower client, start Flower client."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="client")
    parser.add_argument(
        "--num_clients",
        type=check_positive,
        required=True,
    )
    parser.add_argument(
        "--idx",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--staleness_bound",
        type=check_positive,
        required=True,
    )
    parser.add_argument(
        "--min_delay",
        type=check_positive,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--max_delay",
        type=check_positive,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--server_ip",
        type=str,
        default="[::]:8080",
        required=False,
    )
    args = parser.parse_args()
    print(args)

    if args.idx >= args.num_clients:
        print("Usage: idx should be zero-indexed")
        exit(1)

    print(f"I'm client {args.idx}")

    # delay related
    delay = False
    args.max_delay = max(args.max_delay, args.min_delay)
    if args.min_delay > 0:
        delay = True

    if delay:
        print(f"Start SSP client with delay [{args.min_delay},{args.max_delay}]")
    else:
        print("Start SSP client without delay")

    # Load data (CIFAR-10)
    train_loader, test_loader = load_data(args.idx, args.num_clients)

    fl.client.start_numpy_client_ssp(
        args.server_ip,
        CifarClient(MODEL, train_loader, test_loader),
        args.staleness_bound,
        delay=delay,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        switchpoint_strategy=AccuracyVariance(last_k_data=2, var_threshold=0.5, clear_on_switch=False),
    )

# Flower client
class CifarClient(fl.client.NumPyClient):

    def __init__(self, net: model=SimpleCNN(), train_loader=None, test_loader=None):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self):
        return self.net.get_weights()

    def set_parameters(self, parameters):
        self.net.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = train(self.net, self.train_loader, epochs=1)
        return self.get_parameters(), len(self.train_loader), loss, acc

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.test_loader)
        return len(self.test_loader), float(loss), float(accuracy)


def train(net, train_loader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # running_loss and accuracy should record statistics in last epoch
    running_loss = 0
    accuracy = 0
    for _ in range(epochs):
        correct = 0

        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, pred_labels = torch.max(output, dim=1)
            correct += (pred_labels == labels).sum().item()
            running_loss += loss.item()

        accuracy = correct / (len(train_loader) * BATCH_SIZE)

    print(f"loss={running_loss}, acc={accuracy}")
    return running_loss, accuracy


def test(net, test_loader):
    # TODO: The accuracy here seems strange
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    full_test_loader = get_full_test_dataloader()
    with torch.no_grad():
        for data in full_test_loader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Testing on full test set: {len(full_test_loader)}, {loss}, {accuracy}")

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Testing on partial test set: {len(test_loader)}, {loss}, {accuracy}")
    return loss, accuracy


def load_data(idx, num_clients, batch_size=32, noniid=True):
    """Load CIFAR-10 (training and test set)."""
    if noniid:
        # Non-iid
        train_loaders, test_loaders = get_data_loaders(num_clients, batch_size, DATA_FRACTION, 5, False)
        print(u"\u001b[32;1m"
              f"Client {idx}: {len(train_loaders[idx].dataset)} train samples, "
              f"{len(test_loaders[idx].dataset)} test samples"
              u"\u001b[0m")
        return train_loaders[idx], test_loaders[idx]
    else:
        # iid data, all clients use the same dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_set = CIFAR10(".", train=True, download=True, transform=transform)
        test_set = CIFAR10(".", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        return train_loader, test_loader

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == "__main__":
    main()
