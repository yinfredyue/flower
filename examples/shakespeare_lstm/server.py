import flwr as fl
import flwr.common
import torch
from typing import Callable, Dict, Optional, Tuple
from dataset import TEST_DIR, TRAIN_DIR, load_data_iid
from model import RNN
import client

from flwr.server.strategy import FedAvg, Strategy
from flwr.common.switchpoint import TestStrategy, AccuracyVariance
import argparse

def get_eval_fn(num_clients) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = RNN()
        model.set_weights(weights)
        model.to(device)
        _, (x_test, y_test) = load_data_iid(TRAIN_DIR, TEST_DIR, num_clients)
        return client.test(model, x_test, y_test)

    return evaluate


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


# Start Flower server for three rounds of federated learning
# Example: python server.py --num_clients 2 --staleness_bound 2 --rounds 3
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="server")
    parser.add_argument(
        "--num_clients",
        type=check_positive,
        required=True,
    )
    parser.add_argument(
        "--staleness_bound",
        type=check_positive,
        required=True,
    )
    parser.add_argument(
        "--rounds",
        type=check_positive,
        required=True,
    )
    args = parser.parse_args()
    print(args)

    SERVER_ACC_VAR = 1000
    sp_strategy = None
    if args.staleness_bound == SERVER_ACC_VAR:
        sp_strategy = AccuracyVariance(5, 0.001, False)

    print("switchpoint strategy is ", sp_strategy)

    fl.server.start_server_ssp(
        staleness_bound=args.staleness_bound if sp_strategy is None else args.rounds // 2,
        num_clients=args.num_clients,
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        strategy=FedAvg(eval_fn=get_eval_fn(args.num_clients)),
        switchpoint_strategy=sp_strategy,
    )
