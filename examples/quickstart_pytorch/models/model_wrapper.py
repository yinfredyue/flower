from collections import OrderedDict
import flwr as fl

import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self) -> None:
        super(model, self).__init__()

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        # print(self.state_dict().keys())
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights) if 'num_batches_tracked' not in k}
        )

        self.load_state_dict(state_dict, strict=False)
