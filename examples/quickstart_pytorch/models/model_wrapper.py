from collections import OrderedDict
from flwr.common.logger import log
from logging import DEBUG
import os
import flwr as fl

import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self) -> None:
        super(model, self).__init__()
        self.first_get_called = True
        self.model_path = "init_model"

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        if self.first_get_called:
            if os.path.exists(self.model_path):
                self.load_state_dict(torch.load(self.model_path))
                log(DEBUG, "Load existing initial model")
            else:
                torch.save(self.state_dict(), self.model_path)
            self.first_get_called = False
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        # print(self.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights) if 'num_batches_tracked' not in k}
        )

        self.load_state_dict(state_dict, strict=False)
