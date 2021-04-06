import json
import os
from typing import Tuple

import numpy as np
from typing import List

TRAIN_DIR="./shakespeare/data/train/"
TEST_DIR="./shakespeare/data/test/"

"""
data saved from leaf:https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare
using: ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8
   or: ./preprocess.sh -s niid --sf 0.05 -k 10 -t sample -tf 0.8
for full datasets and 0.8 split for train and test data 
and saved in dataset/ 
"""


def read_data(train_data_dir, test_data_dir):
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)

        clients.extend(cdata["users"])

        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])

        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)

        test_data.update(cdata["user_data"])

    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data


def load_data(
    train_data_dir, test_data_dir, client_id: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    clients, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    # Avoid index out of range
    client_id = client_id % len(clients)

    client_name = clients[client_id]
    train_data = train_data[client_name]
    test_data = test_data[client_name]

    x_train = train_data["x"]
    y_train = train_data["y"]
    x_test = test_data["x"]
    y_test = test_data["y"]

    x_train = [word_to_indices(word) for word in x_train]
    x_train = np.array(x_train)
    x_test = [word_to_indices(word) for word in x_test] # One hot encoding
    x_test = np.array(x_test)

    y_train = [letter_to_vec(c) for c in y_train]
    y_train = np.array(y_train)
    y_test = [letter_to_vec(c) for c in y_test] # One hot encoding
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def load_data_iid(train_data_dir, test_data_dir) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    clients, groups, train_data_total, test_data_total = read_data(train_data_dir, test_data_dir)

    n = len(clients)

    x_train_res = []
    y_train_res = []
    x_test_res = []
    y_test_res = []

    for client_name in clients:
        train_data = train_data_total[client_name]
        test_data = test_data_total[client_name]

        ##
        x_train = train_data["x"]
        y_train = train_data["y"]
        x_test = test_data["x"]
        y_test = test_data["y"]

        x_train = [word_to_indices(word) for word in x_train]
        x_train = np.array(x_train)
        x_test = [word_to_indices(word) for word in x_test]  # One hot encoding
        x_test = np.array(x_test)

        y_train = [letter_to_vec(c) for c in y_train]
        y_train = np.array(y_train)
        y_test = [letter_to_vec(c) for c in y_test]  # One hot encoding
        y_test = np.array(y_test)
        ##

        len1 = len(x_train) // n
        x_train_res.extend(x_train[:len1])
        y_train_res.extend(y_train[:len1])

        len2 = len(x_test) // n
        x_test_res.extend(x_test[:len2])
        y_test_res.extend(y_test[:len2])

    return (np.array(x_train_res), np.array(y_train_res)), (np.array(x_test_res), np.array(y_test_res))


# --------------------------------------------------------------------------------
# utils for shakespeare dataset

ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size) -> List[int]:
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter) -> List[int]:
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word) -> List[int]:
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))

    return indices

def one_hot_to_idx(one_hot: List[int]) -> int:
    for i in range(len(one_hot)):
        if one_hot[i] != 0:
            return i

    return -1


if __name__ == '__main__':
    load_data_iid(TRAIN_DIR, TEST_DIR)
