import flwr as fl
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision
from flwr.server.strategy import FedAvg, Strategy
import cifar_test as test
from models import *
from noniid_cifar10 import get_full_testset
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = ResNet18()

def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = MODEL
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return test.test(model, testloader, device=DEVICE)

    return evaluate


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


# Start Flower server for three rounds of federated learning
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

    fl.server.start_server_ssp(
        staleness_bound=args.staleness_bound,
        num_clients=args.num_clients,
        server_address="[::]:8080",
        config={"num_rounds": args.rounds},
        strategy=FedAvg(eval_fn=get_eval_fn(get_full_testset()))
    )
