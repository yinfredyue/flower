from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

import flwr as fl
from flwr.common.switchpoint import TestStrategy, AccuracyVariance
import argparse
from model import RNN
import numpy as np

from flwr.common.chaos import is_straggler
from dataset import load_data, NUM_LETTERS, one_hot_to_idx, TRAIN_DIR, TEST_DIR
from sp_strategy import get_sp_strategy

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
        loss, acc = train(self.model, self.x_train, self.y_train)
        return self.get_parameters(), len(self.x_train), loss, acc

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


def train(model, x_train, y_train, lr=0.002, epochs=1):
    print(f"train: {len(x_train)} samples")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Subsample again, to reduce the amount of train data:
    subsample_rate = 0.01
    # sf=0.05, 0.01
    n = int(len(x_train) * 0.01)
    print(f"n={n}")

    # running_loss and accuracy should record statistics in last epoch
    running_loss = 0
    accuracy = 0
    for _ in range(epochs):
        hidden_state = None
        running_loss = 0

        i = 0
        correct = 0

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

            y_idx = one_hot_to_idx(y)
            pred_char_output = output[len(output)-1]
            pred = torch.argmax(pred_char_output).item()
            correct += (y_idx == pred)

            i += 1
            if i == n:
                break

        accuracy = correct / n

    print(f"loss={running_loss}, acc={accuracy}")
    return running_loss, accuracy


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
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    print(args)
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
    min_delay = 0
    max_delay = max(args.max_delay, args.min_delay)
    if args.min_delay > 0:
        delay = True

    if not delay:
        if is_straggler(args.idx, args.num_clients, 0.5):
            delay = True
            min_delay = 100
            max_delay = 200

    if delay:
        print(f"Start SSP client with delay [{min_delay},{max_delay}]")
    else:
        print("Start SSP client without delay")

    sp_strategy = get_sp_strategy(args.staleness_bound, is_server=False)
    print("switchpoint strategy is ", sp_strategy)

    fl.client.start_numpy_client_ssp(
        args.server_ip,
        ShakespeareClient(args.idx),
        args.staleness_bound if sp_strategy is None else args.rounds // 2,  # Hardcode, we run 30 rounds, set s = 30/2
        delay=delay,
        min_delay=min_delay,
        max_delay=max_delay,
        switchpoint_strategy=sp_strategy,
    )

if __name__ == "__main__":
    # python client.py --num_clients 2 --staleness_bound 2 --idx 0
    # python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.4:8080 --idx 0
    main()
