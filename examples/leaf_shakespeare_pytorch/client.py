from logging import DEBUG, INFO
from typing import Tuple, Dict

import numpy as np

import flwr as fl
from flwr.common.logger import log
import argparse

from model import Model
from fl_utils.model_utils import read_data

def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Shakespeare Client
    class ShakespeareClient(fl.client.NumPyClient):
        def __init__(
            self,
            client_id: int,
            train_data: Dict[str, np.ndarray],
            test_data: Dict[str, np.ndarray],
            model: Model
        ):
            super().__init__()
            self._model = model
            self.cid = client_id
            self.train_data = train_data
            self.test_data = test_data

            # Training parameters
            self.num_epochs = 1
            self.batch_size = 10
            self.lr = None

            log(INFO, u"\u001b[32mClient has %d train samples, %d test samples\u001b[0m",
                len(self.train_data['y']),
                len(self.test_data['y']))

        def get_parameters(self):
            # TODO: Get parameters
            pass

        def fit(self, parameters, config):
            weights: fl.common.Weights = parameters
            updated_parameters = self._model.train(self.train_data, weights)
            num_train_samples = len(self.train_data['y'])
            return updated_parameters, num_train_samples


        def evaluate(self, parameters, config):
            weights: fl.common.Weights = parameters
            return self._model.test(self.test_data, weights)


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


    # load dataset
    # TODO: What is split by user?
    train_data, test_data = read_data("shakespeare/data/train", "shakespeare/data/test")

    client = ShakespeareClient(
        client_id=args.idx,
        train_data=train_data,
        test_data=test_data,
        model=Model(0.0003)
    )

    fl.client.start_numpy_client_ssp(
        args.server_ip,
        client,
        args.staleness_bound,
        delay=delay,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

if __name__ == "__main__":
    main()
