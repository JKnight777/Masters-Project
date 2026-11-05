import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow
import numpy as np
import random
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization

ignore_label = -100

try:
    snr_type = sys.argv[1]
    assert snr_type in ['mixed', 'single_level', 'clean']
except:
    print("Please run with 'single_level', 'mixed', or 'clean'")
    exit(1)

train_dir = Path(f"../data/{snr_type}_data/train/")
test_dir = Path(f"../data/mixed_data/test/")

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

x_train, y_train = load_frames(train_dir)
x_test, y_test = load_frames(test_dir)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Fit normalization using training frames only.
normalizer = Normalization(axis=-1)
normalizer.adapt(x_train)

model = Sequential([
    tf.keras.Input(shape=(x_train.shape[1],)),
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
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')