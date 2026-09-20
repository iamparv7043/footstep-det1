"""
preprocessing/audio.py

Raw-audio level preprocessing shared identically by training and live
inference: load, mono, resample, normalize, fixed-length windowing.
"""

import numpy as np
import librosa


def load_wav_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def normalize_amplitude(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < eps:
        return y
    return y / peak


def to_fixed_length(y: np.ndarray, sr: int, duration_sec: float = 1.0,
                     random_crop: bool = False) -> np.ndarray:
    target_len = int(duration_sec * sr)
    cur_len = len(y)
    if cur_len == target_len:
        return y
    if cur_len < target_len:
        return np.pad(y, (0, target_len - cur_len), mode="constant")
    if random_crop:
        start = np.random.randint(0, cur_len - target_len + 1)
    else:
        start = (cur_len - target_len) // 2
    return y[start:start + target_len]


def preprocess_file(path: str, sr: int = 16000, duration_sec: float = 1.0,
                     random_crop: bool = False) -> np.ndarray:
    y = load_wav_mono(path, target_sr=sr)
    y = normalize_amplitude(y)
    y = to_fixed_length(y, sr, duration_sec, random_crop=random_crop)
    return y


def preprocess_buffer(y: np.ndarray, sr: int = 16000,
                       duration_sec: float = 1.0) -> np.ndarray:
    y = normalize_amplitude(y.astype(np.float32))
    y = to_fixed_length(y, sr, duration_sec, random_crop=False)
    return y
