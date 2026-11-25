import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import sys
import time
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
from collections import deque

SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160
CONTEXT_FRAMES = 5
N_FEATURES = 13

class_map = {
    0:  'sil',
    1:  'PP', 
    2:  'FF', 
    3:  'TH', 
    4:  'DD', 
    5:  'kk', 
    6:  'CH', 
    7:  'SS', 
    8:  'nn', 
    9:  'RR', 
    10: 'aa', 
    11: 'E',  
    12: 'I',  
    13: 'O',  
    14: 'U',  
}

model_path = sys.argv[1]
model = tf.keras.models.load_model(model_path, compile=False)

expected_shape = (CONTEXT_FRAMES, N_FEATURES)
if tuple(model.input_shape[1:]) != expected_shape:
    raise ValueError(
        f"Expected model input (None, {CONTEXT_FRAMES}, {N_FEATURES}), "
        f"got {model.input_shape}"
    )


def extract_one_frame(audio_frame):
    """Calculate one MFCC vector from exactly 400 audio samples."""
    power = librosa.feature.melspectrogram(
        y=audio_frame,
        sr=SAMPLE_RATE,
        n_fft=FRAME_LENGTH,
        win_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=40,
        fmin=20,
        fmax=SAMPLE_RATE / 2,
        window="hann",
        center=False,
        power=2.0,
    )

    log_mel = librosa.power_to_db(
        power,
        ref=1.0,
        amin=1e-10,
        top_db=None,
    )

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=N_FEATURES,
        dct_type=2,
        norm="ortho",
    )

    return mfcc[:, 0].astype(np.float32)


@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=(1, CONTEXT_FRAMES, N_FEATURES),
            dtype=tf.float32,
        )
    ]
)
def infer(x):
    return model(x, training=False)


# Warm Up
extract_one_frame(np.zeros(FRAME_LENGTH, dtype=np.float32))

dummy = tf.zeros((1, CONTEXT_FRAMES, N_FEATURES), dtype=tf.float32)
for _ in range(10):
    infer(dummy).numpy()


class TimingStats:
    """Running statistics without storing every measurement."""
    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0

    def add(self, elapsed_ms):
        self.count += 1
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)

    @property
    def mean_ms(self):
        return self.total_ms / self.count if self.count else 0.0


inference_stats = TimingStats()
processing_stats = TimingStats()

audio_buffer = np.zeros(FRAME_LENGTH, dtype=np.float32)
mfcc_buffer = deque(maxlen=CONTEXT_FRAMES)

p = pyaudio.PyAudio()
stream = None

try:
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        input=True,
        rate=SAMPLE_RATE,
        frames_per_buffer=HOP_LENGTH,
    )

    print("Streaming. Press Ctrl+C to stop.")

    # Initial 25 ms needed for start up
    raw = stream.read(FRAME_LENGTH, exception_on_overflow=True)
    audio_buffer[:] = np.frombuffer(raw, dtype=np.float32)
    mfcc_buffer.append(extract_one_frame(audio_buffer))

    last_report = time.perf_counter()

    while True:
        # Blocking acquisition is deliberately outside processing timing.
        raw = stream.read(HOP_LENGTH, exception_on_overflow=True)
        processing_start = time.perf_counter()

        new_audio = np.frombuffer(raw, dtype=np.float32)

        # Retain 240 old samples and append 160 new samples.
        audio_buffer[:-HOP_LENGTH] = audio_buffer[HOP_LENGTH:]
        audio_buffer[-HOP_LENGTH:] = new_audio

        mfcc_buffer.append(extract_one_frame(audio_buffer))

        if len(mfcc_buffer) < CONTEXT_FRAMES:
            continue

        x = np.stack(mfcc_buffer, axis=0)[None, ...]

        inference_start = time.perf_counter()

        scores = infer(x).numpy()

        inference_end = time.perf_counter()
        class_id = int(np.argmax(scores[0]))
        processing_end = time.perf_counter()

        inference_stats.add(
            (inference_end - inference_start) * 1000
        )
        processing_stats.add(
            (processing_end - processing_start) * 1000
        )

        # Report periodically instead of printing every 10 ms.
        now = time.perf_counter()
        if now - last_report >= 0.5:
            queued_ms = (
                stream.get_read_available() / SAMPLE_RATE * 1000
            )

            print(
                f"Class: {class_map[class_id]:>3} | "
                f"Mean inference: {inference_stats.mean_ms:.2f} ms | "
                f"Mean processing: {processing_stats.mean_ms:.2f} ms | "
                f"Queued audio: ~{queued_ms:.1f} ms"
            )
            last_report = now

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    if stream is not None:
        stream.stop_stream()
        stream.close()
    p.terminate()

    if inference_stats.count:
        print(f"\nPredictions: {inference_stats.count}")
        print(
            f"Inference: mean={inference_stats.mean_ms:.3f} ms, "
            f"max={inference_stats.max_ms:.3f} ms"
        )
        print(
            f"Processing: mean={processing_stats.mean_ms:.3f} ms, "
            f"max={processing_stats.max_ms:.3f} ms"
        )