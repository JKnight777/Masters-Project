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
from tensorflow.keras.layers import Dense, Dropout, Normalization

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
    assert snr_type in ['mixed', 'single_level', 'clean']
    weights = Path(sys.argv[2])
    # assert snr_type in weights
    data_dir = Path(sys.argv[3])
    # run = sys.argv[4]
except:
    print("Please run with 'single_level', 'mixed', or 'clean'")
    exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'{weights.parent}/{snr_type}_model_test.log', level=logging.INFO)

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

    x_train = np.concatenate(features, axis=0)
    y_train = np.concatenate(labels, axis=0)
    return x_train, y_train

train_dir = Path(data_dir / f'{snr_type}_data' / 'train')
test_dir = Path(data_dir / f'mixed_data' / 'test')
print(train_dir)
x_train, y_train = load_frames(train_dir)
x_test, y_test = load_frames(test_dir)

# Fit normalization using training frames only.
normalizer = Normalization(axis=-1)
normalizer.adapt(x_train)

logging.info(f"Training {snr_type} model. Model is made up of 2 Dense hidden layers, 128 -> 64. Both activated using Relu")
model = Sequential([
    tf.keras.Input(shape=(x_test.shape[1],)),
    normalizer,
    Dense(128, activation='relu'), 
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(15)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [SparseF1Score(num_classes=15, name="macro_f1"), tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)

# history = model.fit(
#     x_train,
#     y_train,
#     epochs=75,
#     batch_size=256,
#     shuffle=True,
#     validation_split=0.2,
#     verbose=1,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         )
#     ]
# )

# model.save_weights(f'{snr_type}_model.weights.h5')
# model.save(f'{snr_type}_model.keras')

model.load_weights(weights)

# model.summary()
# model.summary(print_fn=logging.info)

logging.info("\nTest Results:")
loss, macro_f1, accuracy = model.evaluate(x_test, y_test)
print(f'Loss: {loss}')
print(f'Macro F1: {macro_f1 * 100:.2f}%')
print(f'Accuracy: {accuracy * 100:.2f}%')


logging.info(f'Loss: {loss}')
logging.info(f'Macro F1: {macro_f1 * 100:.2f}%')
logging.info(f'Accuracy: {accuracy * 100:.2f}%\n')
