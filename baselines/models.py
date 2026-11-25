import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
import numpy as np
import random
import sys
import logging
from pathlib import Path
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Flatten

CONTEXT_FRAMES = 5
NUM_FEATURES = 13
INPUT_SHAPE = (NUM_FEATURES,) # (CONTEXT_FRAMES, NUM_FEATURES) # for Frame Windows

@keras.saving.register_keras_serializable()
class SparseF1Score(tf.keras.metrics.F1Score):
    def __init__(self, num_classes=15, average="macro", threshold=None, name="sparse_f1_score", dtype=None, **kwargs):
        super().__init__(average=average, threshold=threshold, name=name, dtype=dtype, **kwargs)
        self.num_classes = int(num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert integer labels to one-hot vectors
        y_true = tf.reshape(y_true, [-1])
        y_true = tf.one_hot(
            tf.cast(y_true, tf.int32),
            depth=self.num_classes,
            dtype=self.dtype
        )

        return super().update_state(
            y_true, y_pred, sample_weight=sample_weight
        )

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

def mlp_model(normalizer):
    return Sequential([
        tf.keras.Input(shape=INPUT_SHAPE),
        normalizer,
        Flatten(),
        Dense(128, activation='relu'), 
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(15)
    ])

def c1dnn_model(normalizer):
    return Sequential([
        tf.keras.Input(shape=INPUT_SHAPE),
        normalizer,
        tf.keras.layers.Conv1D(
            filters = 32,
            kernel_size = 3,
            padding = 'valid',
            activation = 'relu',
        ),
        tf.keras.layers.Conv1D(
            filters = 64,
            kernel_size = 3,
            padding = 'valid',
            activation = 'relu',
        ),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(15)
    ])