"""
dataset.py

Scans dataset/footstep and dataset/no_footstep, splits by SOURCE FILE
(not by chunk) into train/val/test to prevent leakage, and returns
(log_mel_tensor, aux_feature_tensor, label) triples.
"""

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.audio import load_wav_mono, normalize_amplitude, to_fixed_length
from preprocessing.features import (
    compute_log_mel_spectrogram, compute_low_freq_ratio, spec_augment, DEFAULT_PARAMS,
)

LABEL_NAMES = ["no_footstep", "footstep"]


@dataclass
class Segment:
    source_path: str
    label: int
    start_sample: int


def _list_wavs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.wav")) +
                  glob.glob(os.path.join(folder, "*.WAV")))


def build_segment_index(dataset_root: str, sr: int, duration_sec: float,
                         hop_sec: float = 0.5) -> List[Segment]:
    import librosa
    segments = []
    for label, cls in enumerate(LABEL_NAMES):
        folder = os.path.join(dataset_root, cls)
        for f in _list_wavs(folder):
            try:
                dur = librosa.get_duration(path=f)
            except Exception:
                continue
            n_samples = int(dur * sr)
            win = int(duration_sec * sr)
            hop = int(hop_sec * sr)
            if n_samples <= win:
                segments.append(Segment(f, label, 0))
            else:
                start = 0
                while start < n_samples - 1:
                    segments.append(Segment(f, label, start))
                    start += hop
                    if start + win > n_samples:
                        break
    return segments


def split_by_source_file(segments: List[Segment], seed: int = 42,
                          train_frac: float = 0.70,
                          val_frac: float = 0.15) -> Tuple[List[Segment], List[Segment], List[Segment]]:
    rng = random.Random(seed)
    files_by_class = {0: set(), 1: set()}
    for s in segments:
        files_by_class[s.label].add(s.source_path)

    train_files, val_files, test_files = set(), set(), set()
    for label, files in files_by_class.items():
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_frac)) if n > 2 else n
        n_val = max(1, int(n * val_frac)) if n > 2 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        train_files.update(files[:n_train])
        val_files.update(files[n_train:n_train + n_val])
        test_files.update(files[n_train + n_val:])

    train_segs = [s for s in segments if s.source_path in train_files]
    val_segs = [s for s in segments if s.source_path in val_files]
    test_segs = [s for s in segments if s.source_path in test_files]
    return train_segs, val_segs, test_segs


def add_background_noise(y: np.ndarray, noise_level: float) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32)
    return y + noise_level * noise


def random_gain(y: np.ndarray, low_db: float = -6.0, high_db: float = 6.0) -> np.ndarray:
    gain_db = np.random.uniform(low_db, high_db)
    return y * (10.0 ** (gain_db / 20.0))


def random_time_shift(y: np.ndarray, sr: int, max_shift_sec: float = 0.1) -> np.ndarray:
    max_shift = int(max_shift_sec * sr)
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(y, shift)


class FootstepDataset(Dataset):
    def __init__(self, segments: List[Segment], sr: int = 16000,
                 duration_sec: float = 1.0, mel_params: dict = None,
                 augment: bool = False):
        self.segments = segments
        self.sr = sr
        self.duration_sec = duration_sec
        self.mel_params = mel_params or DEFAULT_PARAMS
        self.augment = augment
        self._audio_cache = {}

    def __len__(self):
        return len(self.segments)

    def _load_source(self, path: str) -> np.ndarray:
        if path not in self._audio_cache:
            y = load_wav_mono(path, target_sr=self.sr)
            y = normalize_amplitude(y)
            self._audio_cache[path] = y
        return self._audio_cache[path]

    def __getitem__(self, idx):
        seg = self.segments[idx]
        full = self._load_source(seg.source_path)
        win = int(self.duration_sec * self.sr)
        chunk = full[seg.start_sample: seg.start_sample + win]
        chunk = to_fixed_length(chunk, self.sr, self.duration_sec, random_crop=False)

        if self.augment:
            chunk = random_gain(chunk)
            chunk = random_time_shift(chunk, self.sr)
            chunk = add_background_noise(chunk, noise_level=np.random.uniform(0.0, 0.01))
            chunk = np.clip(chunk, -1.0, 1.0)

        low_freq_ratio = compute_low_freq_ratio(chunk, sample_rate=self.sr)

        log_mel = compute_log_mel_spectrogram(
            chunk,
            sample_rate=self.mel_params["sample_rate"],
            n_fft=self.mel_params["n_fft"],
            hop_length=self.mel_params["hop_length"],
            win_length=self.mel_params["win_length"],
            n_mels=self.mel_params["n_mels"],
        )
        if self.augment:
            log_mel = spec_augment(log_mel)

        tensor = torch.from_numpy(log_mel).unsqueeze(0)
        aux = torch.tensor([low_freq_ratio], dtype=torch.float32)
        return tensor, aux, seg.label


def class_counts(segments: List[Segment]) -> dict:
    counts = {0: 0, 1: 0}
    for s in segments:
        counts[s.label] += 1
    return counts
