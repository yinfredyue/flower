import flwr as fl
import flwr.common
from typing import Callable, Dict, Optional, Tuple

from flwr.server.strategy import FedAvg, Strategy
import argparse

def get_eval_fn(

) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        # TODO Evaluate weights and return
        lss, acc = 0.1, 0.1

        return lss, acc

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
        config={"num_rounds": 30},
        strategy=FedAvg(eval_fn=get_eval_fn())
    )
