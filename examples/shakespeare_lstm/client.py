from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

import flwr as fl
import argparse
from model import RNN
import numpy as np

from dataset import load_data, NUM_LETTERS, one_hot_to_idx, TRAIN_DIR, TEST_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flower client
class ShakespeareClient(fl.client.NumPyClient):
    def __init__(self, idx, lr=0.001):
        self.model = RNN()
        self.lr = lr
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_data(TRAIN_DIR, TEST_DIR, idx)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.x_train, self.y_train)
        return self.get_parameters(), len(self.x_train)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.x_test, self.y_test)
        return len(self.x_test), float(loss), float(accuracy)


def to_tensor(raw):
    return torch.LongTensor(raw).to(device)


def get_target_seq(x, y) -> torch.Tensor:
    y_idx = one_hot_to_idx(y)
    seq = x[1:]
    seq = np.append(seq, y_idx)
    seq = torch.tensor(seq)
    return seq


def train(model, x_train, y_train, lr=0.002, epochs=1, n=300):
    print(f"train: {len(x_train)} samples")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for _ in range(epochs):
        hidden_state = None
        running_loss = 0

        i = 0
        for x, y in zip(x_train, y_train):
            target_seq = get_target_seq(x, y)
            x = to_tensor(x)
            x = torch.unsqueeze(x, dim=1)  # [80] -> [80, 1]
            output, hidden_state = model(x, hidden_state)

            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i % 100 == 0:
                print(f"train: done {i}")
            if i == n:
                return



def test(model, x_test, y_test):
    print(f"test: {len(x_test)} samples")

    running_loss = 0
    correct = 0

    hidden_state = None
    loss_fn = nn.CrossEntropyLoss()

    for x, y in zip(x_test, y_test):
        y_idx = one_hot_to_idx(y)
        x = to_tensor(x)
        x = torch.unsqueeze(x, dim=1)  # [80] -> [80, 1]

        with torch.no_grad():
            target_seq = get_target_seq(x, y)
            output, hidden_state = model(x, hidden_state)

            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            pred_char_output = output[len(output) - 1]
            pred = torch.argmax(pred_char_output).item()

            if pred == y_idx:
                correct += 1

    return running_loss, correct/len(x_test)


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def parse_args(parser: argparse.ArgumentParser):
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
    return args


def main():
    # python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.4:8080 --idx 0
    """Create model, load data, define Flower client, start Flower client."""

    parser = argparse.ArgumentParser(description="client")
    args = parse_args(parser)

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

    fl.client.start_numpy_client_ssp(
        args.server_ip,
        ShakespeareClient(args.idx),
        args.staleness_bound,
        delay=delay,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )

if __name__ == "__main__":
    # python client.py --num_clients 2 --staleness_bound 2 --idx 0
    # python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.4:8080 --idx 0
    main()
