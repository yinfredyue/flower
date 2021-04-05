# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import time
import timeit
from logging import INFO
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import flwr
from flwr.common.logger import log

tf.enable_eager_execution()

def custom_fit(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_epochs: int,
    batch_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
    delay_factor: float = 0.0,
    timeout: Optional[int] = None,
) -> Tuple[bool, float, int]:
    """Train the model using a custom training loop."""
    ds_train = dataset.batch(batch_size=batch_size, drop_remainder=False)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    fit_begin = timeit.default_timer()
    num_examples = 0
    for epoch in range(num_epochs):
        log(INFO, "Starting epoch %s", epoch)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Single loop over the dataset
        batch_begin = timeit.default_timer()
        for x, y in ds_train:
            # Optimize the model
            loss_value, grads = grad(model, x, y) # Compute loss and gradient
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add the current batch loss
            epoch_accuracy.update_state(y, model(x, training=False))

            # Track the number of examples used for training
            num_examples += x.shape[0]

            # Delay
            batch_duration = timeit.default_timer() - batch_begin
            if delay_factor > 0.0:
                time.sleep(batch_duration * delay_factor)
            if timeout is not None:
                fit_duration = timeit.default_timer() - fit_begin
                if fit_duration > timeout:
                    log(INFO, "client timeout")
                    return (False, fit_duration, num_examples)
            batch_begin = timeit.default_timer()

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        log(
            INFO,
            u"\u001b[32;1mEpoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}\u001b[0m".format(
                epoch, epoch_loss_avg.result(), epoch_accuracy.result()
            ),
        )

    fit_duration = timeit.default_timer() - fit_begin
    return True, fit_duration, num_examples


def loss(
    model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor, training: bool
) -> tf.Tensor:
    """Calculate categorical cross-entropy loss."""
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def grad(
    model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Calculate gradients."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def keras_evaluate(
    model: tf.keras.Model, dataset: tf.data.Dataset, batch_size: int
) -> Tuple[float, float]:
    """Evaluate the model using model.evaluate(...)."""
    ds_test = dataset.batch(batch_size=batch_size, drop_remainder=False)
    test_loss, acc = model.evaluate(x=ds_test)
    return float(test_loss), float(acc)


def keras_fit(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_epochs: int,
    batch_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
) -> None:
    """Train the model using model.fit(...)."""
    ds_train = dataset.batch(batch_size=batch_size, drop_remainder=False)
    model.fit(ds_train, epochs=num_epochs, callbacks=callbacks, verbose=2)


def get_lr_schedule(
    epoch_global: int, lr_initial: float, lr_decay: float
) -> Callable[[int], float]:
    """Return a schedule which decays the learning rate after each epoch."""

    def lr_schedule(epoch: int) -> float:
        """Learning rate schedule."""
        epoch += epoch_global
        return lr_initial * lr_decay ** epoch

    return lr_schedule


def get_eval_fn(
    model: tf.keras.Model, num_classes: int, xy_test: Tuple[np.ndarray, np.ndarray]
) -> Callable[[flwr.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    ds_test = build_dataset(
        xy_test[0],
        xy_test[1],
        num_classes=num_classes,
        shuffle_buffer_size=0,
        augment=False,
    )

    def evaluate(weights: flwr.common.Weights) -> Optional[Tuple[float, float]]:
        """Use entire test set for evaluation."""
        model.set_weights(weights)
        lss, acc = keras_evaluate(model, ds_test, batch_size=len(xy_test[0]))
        return lss, acc

    return evaluate


def build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    shuffle_buffer_size: int = 0,
    augment: bool = False,
    augment_color: bool = False,
    augment_horizontal_flip: bool = False,
    augment_offset: int = 0,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Divide images by 255, one-hot encode labels, optionally shuffle and augment."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    if augment:
        dataset = dataset.map(
            lambda x, y: (
                apply_augmentation(
                    x,
                    seed=seed,
                    color=augment_color,
                    horizontal_flip=augment_horizontal_flip,
                    offset=augment_offset,
                ),
                y,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    return dataset


def apply_augmentation(
    img: tf.Tensor,
    seed: Optional[int],
    color: bool,
    horizontal_flip: bool,
    offset: int,
) -> tf.Tensor:
    """Apply different augmentations to a single example."""
    if color:
        img = tf.image.random_hue(img, 0.08, seed=seed)
        img = tf.image.random_saturation(img, 0.6, 1.6, seed=seed)
        img = tf.image.random_brightness(img, 0.05, seed=seed)
        img = tf.image.random_contrast(img, 0.7, 1.3, seed=seed)
    if horizontal_flip:
        img = tf.image.random_flip_left_right(img, seed=seed)
    # Get image size from tensor
    size = img.shape.as_list()  # E.g., [28, 28, 1] or [32, 32, 3]
    height = size[0]
    width = size[1]
    img_padded = tf.image.pad_to_bounding_box(
        img, offset, offset, height + 2 * offset, width + 2 * offset
    )
    return tf.image.random_crop(img_padded, size=size, seed=seed)


def stacked_lstm(
    input_len, hidden_size: int, num_classes: int, embedding_dim: int
) -> tf.keras.Model:
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform()

    # Architecture
    inputs = tf.keras.layers.Input(shape=(input_len,))

    # What's an embedding layer:
    # It's the Word2Vec matrix which maps words into vector space.
    # input_dim is the dimension of one-hot encoding;
    # output_dim is the dimension of vector space.
    embedding = tf.keras.layers.Embedding(
        input_dim=num_classes, output_dim=embedding_dim
    )(inputs)
    lstm = tf.keras.layers.LSTM(units=hidden_size, return_sequences=True)(embedding)
    lstm = tf.keras.layers.LSTM(units=hidden_size)(lstm)
    outputs = tf.keras.layers.Dense(
        num_classes, kernel_initializer=kernel_initializer, activation="softmax"
    )(lstm)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model
