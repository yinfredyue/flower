from collections import OrderedDict
import torch
import torch.nn as nn
from dataset import NUM_LETTERS
import flwr as fl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size=NUM_LETTERS, output_size=NUM_LETTERS, hidden_size=512, num_layers=1, rnn_type="gru", drop_prob=0.5):

        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(input_size, input_size)

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        self.decoder = nn.Linear(hidden_size, output_size)


    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)

        # https://stackoverflow.com/a/48278089/9057530
        if self.rnn_type == "gru":
            return output, hidden_state.detach()
        else:
            return output, (hidden_state[0].detach(), hidden_state[1].detach())

    def get_weights(self) -> fl.common.Weights:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)
