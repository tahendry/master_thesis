"""
check if GPU is available
"""
# %% Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.python.platform import build_info as tf_build_info
import numpy as np
import matplotlib.pyplot as plt
import time


# %% Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices("CPU")))
print("Num TPUs Available: ", len(tf.config.experimental.list_physical_devices("TPU")))

print(f"Your cuDNN version is: {tf_build_info.build_info['cudnn_version']}")
print(f"Your CUDA version is: {tf_build_info.build_info['cuda_version']}")

# %% Load the data

(train_data, train_labels), _ = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


train_data = vectorize_sequences(train_data)


# %% Define the model
with tf.device("cpu"):

    model = keras.Sequential(
        [
            layers.Dense(320, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    start = time.time()
    history_dropout = model.fit(
        train_data,
        train_labels,
        epochs=100,
        batch_size=512,
        validation_split=0.4,
    )
    end = time.time()
    time_cpu = end - start


with tf.device("gpu"):
    model = keras.Sequential(
        [
            layers.Dense(320, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(160, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    start = time.time()
    history_dropout = model.fit(
        train_data,
        train_labels,
        epochs=100,
        batch_size=512,
        validation_split=0.4,
    )
    end = time.time()
    time_gpu = end - start


# %% Plot the results
# val_loss = history_dropout.history["val_loss"]
# epochs = range(1, 21)
# plt.plot(epochs, val_loss, "b--", label="Validation loss")
# plt.title("Effect of insufficient model capacity on validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

print("Time CPU: ", time_cpu)
print("Time GPU: ", time_gpu)
