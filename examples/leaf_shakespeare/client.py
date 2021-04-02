from logging import DEBUG
from typing import Tuple

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common.logger import log
from load_data import load_data
from tf_utils import build_dataset, custom_fit, keras_evaluate, stacked_lstm
import argparse


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Shakespeare Client
    class ShakespeareClient(fl.client.NumPyClient):
        def __init__(
            self,
            cid: str,
            model: tf.keras.Model,
            xy_train: Tuple[np.ndarray, np.ndarray],
            xy_test: Tuple[np.ndarray, np.ndarray],
            delay_factor: float,
            num_classes: int,
        ):
            super().__init__(cid)
            self.model = model
            self.cid = cid

            self.ds_train = build_dataset(
                xy_train[0],
                xy_train[1],
                num_classes=num_classes,
                shuffle_buffer_size=len(xy_train[0]),
                augment=False,
            )
            self.ds_test = build_dataset(
                xy_test[0],
                xy_test[1],
                num_classes=num_classes,
                shuffle_buffer_size=0,
                augment=False,
            )

            self.num_examples_train = len(xy_train[0])
            self.num_examples_test = len(xy_test[0])
            self.delay_factor = delay_factor

        def get_parameters(self):
            return self.model.get_weights()

        def fit(self, parameters, config):
            weights: fl.common.Weights = parameters

            # Read training configuration
            epoch_global = int(config["epoch_global"])
            epochs = int(config["epochs"])
            batch_size = int(config["batch_size"])
            timeout = int(config["timeout"])
            partial_updates = bool(int(config["partial_updates"]))

            # Use provided weights to update the local model
            self.model.set_weights(weights)

            # train model on local dataset
            completed, fit_duration, num_examples = custom_fit(
                model=self.model,
                dataset=self.ds_train,
                num_epochs=epochs,
                batch_size=batch_size,
                callbacks=[],
                delay_factor=self.delay_factor,
                timeout=timeout,
            )
            log(DEBUG, "client %s had fit_duration %s", self.cid, fit_duration)

            # Return empty update if local update could not be completed in time
            if not completed and not partial_updates:
                updated_parameters = fl.common.weights_to_parameters([])
                return updated_parameters, num_examples

            # Return the refined weights and the number of examples used for training
            updated_parameters = fl.common.weights_to_parameters(self.model.get_weights())
            return updated_parameters, num_examples

        def evaluate(self, parameters, config):
            weights: fl.common.Weights = parameters

            log(
                DEBUG,
                "evaluate on %s (examples: %s), config %s",
                self.cid,
                self.num_examples_test,
                config,
            )

            self.model.set_weights(weights)

            loss, acc = keras_evaluate(
                self.model, self.ds_test, batch_size=self.num_examples_test
            )

            return self.num_examples_test, loss, acc

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
    xy_train, xy_test = load_data(
        "dataset/data/train",
        "dataset/data/test",
        args.idx,
    )

    model = stacked_lstm(input_len=80, hidden_size=256, num_classes=80, embedding_dim=80)

    client = ShakespeareClient(
        args.idx, model, xy_train, xy_test, 0.0, 80
    )

    fl.client.start_numpy_client_ssp(
        "[::]:8080",
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
