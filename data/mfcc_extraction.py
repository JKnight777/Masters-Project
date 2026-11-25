import librosa
import sys
import json
import numpy as np
from pathlib import Path

FRAME = 400
HOP = 160
CONTEXT_FRAMES = 5

class_map = {
    'sil':0,
    'PP': 1,
    'FF': 2,
    'TH': 3,
    'DD': 4,
    'kk': 5,
    'CH': 6,
    'SS': 7,
    'nn': 8,
    'RR': 9,
    'aa': 10,
    'E':  11,
    'I':  12,
    'O':  13,
    'U':  14
}

def extract_mfccs_fw(wav_file, phn_script):
    audio, sr = librosa.load(wav_file, sr=None)

    power = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft = FRAME, win_length = FRAME, hop_length=HOP, n_mels = 40, fmin=20, fmax = sr / 2, window = "hann", center=False, power=2.0) # fmin could be set to 0 for softer voices

    log_mel = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)

    x = librosa.feature.mfcc(S=log_mel, n_mfcc=13, dct_type=2, norm="ortho").T.astype(np.float32)

    segments = []

    with open(phn_script) as f:
        for line in f:
            start, end, phn = line.split()
            segments.append((int(start), int(end), phn))

    labels = (np.arange(len(x), dtype=np.int64) * HOP + FRAME // 2)
    
    y = np.full(len(x), -100, dtype=np.int64)

    for start, end, phn in segments:
        phn_id = class_map[phn]
        selections = (labels >= start) & (labels < end)
        y[selections] = phn_id

    n_windows = len(x) - CONTEXT_FRAMES + 1

    x_windows = np.stack(
        [x[offset:offset + n_windows] for offset in range(CONTEXT_FRAMES)],
        axis=1,
    )

    y_windows = y[CONTEXT_FRAMES - 1:]
    label_samples = labels[CONTEXT_FRAMES - 1:]

    config = {
        'sr': sr,
        'frame_length': FRAME,
        'hop_length': HOP,
        'n_mels': 40,
        'n_mfcc': 13,
        'center': False,
        'alignment': 'midpoint',
        'context_frame': CONTEXT_FRAMES,
        'context_stride': 1,
        'target_frame': 'last'
    }

    return x_windows, y_windows, label_samples, np.array(json.dumps(config))

def extract_mfccs_sf(wav_file, phn_script):
    audio, sr = librosa.load(wav_file, sr=None)

    power = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft = FRAME, win_length = FRAME, hop_length=HOP, n_mels = 40, fmin=20, 
                                           fmax = sr / 2, window = "hann", center=False, power=2.0)

    log_mel = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)

    x = librosa.feature.mfcc(S=log_mel, n_mfcc=13, dct_type=2, norm="ortho").T.astype(np.float32)

    segments = []

    with open(phn_script) as f:
        for line in f:
            start, end, phn = line.split()
            segments.append((int(start), int(end), phn))

    labels = (np.arange(len(x), dtype=np.int64) * HOP + FRAME // 2)
    
    y = np.full(len(x), -100, dtype=np.int64)

    for start, end, phn in segments:
        phn_id = class_map[phn]
        selections = (labels >= start) & (labels < end)
        y[selections] = phn_id

    config = {
        'sr': sr,
        'frame_length': FRAME,
        'hop_length': HOP,
        'n_mels': 40,
        'n_mfcc': 13,
        'center': False,
        'alignment': 'midpoint'
    }

    return x, y, labels, np.array(json.dumps(config))

