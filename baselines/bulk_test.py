import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
import numpy as np
import random
import sys
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Flatten
from models import mlp_model, c1dnn_model

pyseed = random.randrange(sys.maxsize)
random.seed(pyseed)

tfseed = random.randint(0, 2**32 - 1)
tf.random.set_seed(tfseed)

class SparseF1Score(tf.keras.metrics.F1Score):
    def __init__(self, num_classes=15, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert integer labels to one-hot vectors
        y_true = tf.reshape(y_true, [-1])
        y_true = tf.one_hot(
            tf.cast(y_true, tf.int32),
            depth=self.num_classes
        )

        return super().update_state(
            y_true, y_pred, sample_weight=sample_weight
        )

ignore_label = -100

try:
    snr_type = sys.argv[1]
    assert snr_type in ['mixed', 'clean'] or 'single_level' in snr_type
    weights_dir = Path(sys.argv[2])
    # assert snr_type in weights
    data_dir = Path("C:/Users/Justice/Desktop/Master's Project/data") # Path(sys.argv[3])
    if len(sys.argv) > 3:
        snr_level = sys.argv[3]
    else:
        snr_level = snr_type
    # run = sys.argv[4]
except:
    print("Please run with 'single_level', 'mixed', or 'clean'")
    exit(1)

weights = [weight for weight in weights_dir.rglob('*.weights.h5')]

total_mf1 = 0.0
total_acc = 0.0

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{weights_dir}/{weights_dir.name}_model_on_{snr_level}_test.log', level=logging.INFO)

logging.info(f"Python Random Seed: {pyseed}")
logging.info(f"Tensorflow Seed: {tfseed}")

def load_frames(source_dir):
    npz_files = list(source_dir.rglob('*.npz'))
    random.shuffle(npz_files)
    features = []
    labels = []
    for file in npz_files:
        with np.load(file, allow_pickle=False) as data:
            x = np.asarray(data['x'], dtype=np.float32)
            y = np.asarray(data['y'])

            keep = y != ignore_label
            x = x[keep]
            y = y[keep]
            if len(y):
                features.append(x)
                labels.append(y.astype(np.int32, copy=False))

    x_set = np.concatenate(features, axis=0)
    y_set = np.concatenate(labels, axis=0)
    return x_set, y_set

for weight in weights:
    train_dir = Path(data_dir / f'{snr_level}_data_01')
    test_dir = Path(data_dir / 'Single Frames Test Sets' / f'{snr_level}')
    x_train, y_train = load_frames(train_dir)
    x_test, y_test = load_frames(test_dir)

    logging.info(f'\nWeight File: {weight}')
    # Fit normalization using training frames only.
    normalizer = Normalization(axis=-1)
    normalizer.adapt(x_train)

    logging.info(f"Training {snr_type} model. Model is made up of 2 Dense hidden layers, 128 -> 64. Both activated using Relu")
    # model = Sequential([
    #     tf.keras.Input(shape=x_train.shape[1:]),
    #     normalizer,
    #     Flatten(),
    #     Dense(128, activation='relu'), 
    #     Dropout(0.1),
    #     Dense(64, activation='relu'),
    #     Dropout(0.1),
    #     Dense(15)
    # ])

    model = mlp_model(normalizer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = [SparseF1Score(num_classes=15, name="macro_f1"), tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    model.load_weights(weight)

    logging.info("\nTest Results:")
    loss, macro_f1, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}')
    print(f'Macro F1: {macro_f1 * 100:.2f}%')
    print(f'Accuracy: {accuracy * 100:.2f}%')


    logging.info(f'Loss: {loss}')
    logging.info(f'Macro F1: {macro_f1 * 100:.2f}%')
    logging.info(f'Accuracy: {accuracy * 100:.2f}%\n')

    total_mf1 += macro_f1
    total_acc += accuracy

total_mf1 /= len(weights)
total_acc /= len(weights)

logging.info(f'Average Macro F1: {total_mf1 * 100:.2f}%')
logging.info(f'Average Accuracy: {total_acc * 100:.2f}%\n')

print(f'\nAverage Accuracy: {total_acc * 100:.2f}%')
print(f'Average Macro F1: {total_mf1 * 100:.2f}%\n')