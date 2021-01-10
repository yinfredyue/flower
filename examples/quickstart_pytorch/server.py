import flwr as fl
from typing import Callable, Dict, Optional, Tuple

import torch
import torchvision
from flwr.server.strategy import FedAvg, Strategy
import cifar_test as test

def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = test.load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        return test.test(model, testloader, device=DEVICE)

    return evaluate

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server_ssp(
        server_address="[::]:8080", 
        config={"num_rounds": 20}, 
        strategy=FedAvg(eval_fn=get_eval_fn(test.testset()))
    )