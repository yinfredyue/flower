# Reference: https://github.com/nikhilbarhate99/Char-RNN-PyTorch/blob/master/CharRNN.py

from model import RNN
import torch
import torch.nn as nn
from dataset import load_data, NUM_LETTERS, one_hot_to_idx
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(raw):
    return torch.LongTensor(raw).to(device)

# How CrossEntropyLoss works:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# CrossEntropyLoss takes two inputs of different size.
# In the 1st parameter, each item represents probabilities (a vector of numbers);
# In the 2nd parameter, each item represents the correct result (one number).
#
# loss = nn.CrossEntropyLoss()
#
# input = torch.randn(3, 5, requires_grad=True)
# print(input.size(), input)
#
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target.size(), target)
#
# output = loss(input, target)
# print(output.size(), output)
# output.backward()
#
# print(output.item())

def train_leaf():
    # Hyper-parameters of RNN
    hidden_size = 512  # size of hidden state
    num_layers = 1  # num of layers in LSTM layer stack
    lr = 0.002  # learning rate

    (x_train, y_train), (x_test, y_test) = load_data("./shakespeare/data/train/", "./shakespeare/data/test/", 0)
    print(f"Dataset loaded. Train size: {len(x_train)}, test size: {len(x_test)}")

    rnn = RNN(NUM_LETTERS, NUM_LETTERS, hidden_size, num_layers)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr)

    for epoch in range(10):
        hidden_state = None
        running_loss = 0
        n = 0

        for i in range(min(len(x_train), 200)):
            # x = "Hello worl" [80 characters]
            # y = 'd'          [1 character]
            # target_seq = "ello world" [80 character]

            x = x_train[i]
            y = y_train[i]

            y_idx = one_hot_to_idx(y)
            target_seq = x[1:]
            target_seq = np.append(target_seq, y_idx)
            target_seq = torch.tensor(target_seq)

            x = to_tensor(x)

            x = torch.unsqueeze(x, dim=1)

            output, hidden_state = rnn(x, hidden_state)

            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch, running_loss / n))

        # evalute using test set
        correct = 0
        wrong = 0
        for i in range(len(x_test)):
            # Use test set
            x = x_test[i]
            y = y_test[i]

            y_idx = one_hot_to_idx(y)

            x = to_tensor(x)
            x = torch.unsqueeze(x, dim=1)

            with torch.no_grad():
                # `output` contains the entire sequence of character predication
                output, _ = rnn(x, hidden_state)

                # `pred_char_output` is the last item, which is the next character
                pred_char_output = output[len(output)-1]

                #
                pred = torch.argmax(pred_char_output).item()

                if pred == y_idx:
                    correct += 1
                else:
                    wrong += 1

        print(f"correct: {correct}, wrong: {wrong}, acc: {correct/(correct+wrong)}")

if __name__ == '__main__':
    train_leaf()


