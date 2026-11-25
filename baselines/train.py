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
from models import mlp_model, c1dnn_model, SparseF1Score

pyseed = random.randrange(sys.maxsize)
random.seed(pyseed)

tfseed = random.randint(0, 2**32 - 1)
tf.random.set_seed(tfseed)

ignore_label = -100

try:
    snr_type = sys.argv[1]
    assert snr_type in ['mixed', 'single_level', 'clean']
    run = sys.argv[2]
    if snr_type == 'single_level':
        SINGLE_LEVEL = sys.argv[3]
    else:
        SINGLE_LEVEL = None
except:
    print("Please run with 'single_level', 'mixed', or 'clean'")
    exit(1)

logger = logging.getLogger(__name__)



if snr_type == 'single_level':
    os.makedirs(f'./{SINGLE_LEVEL}', exist_ok=True)
    logging.basicConfig(filename=f'./{SINGLE_LEVEL}/single_level_{SINGLE_LEVEL}_model.log', level=logging.INFO)
else:
    os.makedirs(f'./{snr_type}', exist_ok=True)
    logging.basicConfig(filename=f'./{snr_type}/{snr_type}_model.log', level=logging.INFO)

logging.info(f"Python Random Seed: {pyseed}")
logging.info(f"Tensorflow Seed: {tfseed}")

if SINGLE_LEVEL:
    train_dir = Path(f"../data/{SINGLE_LEVEL}_data_{run}/")
else:
    train_dir = Path(f"../data/{snr_type}_data_{run}/")
# test_dir = Path(f"../data/mixed_data/test/")

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

print(x_train.shape, y_train.shape)

# Fit normalization using training frames only.
normalizer = Normalization(axis=-1)
normalizer.adapt(x_train)
print(x_train[0].shape)
exit()
model = c1dnn_model(normalizer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [SparseF1Score(num_classes=15, name="macro_f1"), tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)

history = model.fit(
    x_train,
    y_train,
    epochs=75,
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

if snr_type == 'single_level':
    model.save_weights(f'./{SINGLE_LEVEL}/{SINGLE_LEVEL}_model_{run}.weights.h5')
    model.save(f'./{SINGLE_LEVEL}/{SINGLE_LEVEL}_model_{run}.keras')
else:
    model.save_weights(f'./{snr_type}/{snr_type}_model_{run}.weights.h5')
    model.save(f'./{snr_type}/{snr_type}_model_{run}.keras')

model.summary()
model.summary(print_fn=logging.info)