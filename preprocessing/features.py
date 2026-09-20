"""
preprocessing/features.py

Log-Mel Spectrogram extraction (identical for training/eval/live), a
low-frequency energy ratio auxiliary feature (footsteps are low-freq
thumps; speech/claps/etc are broadband/high-freq -- this feature helps
the model, and infer_live.py, tell them apart), and SpecAugment
(training-only augmentation on the spectrogram itself).
"""

import numpy as np
import librosa

DEFAULT_PARAMS = dict(
    sample_rate=16000,
    n_fft=1024,
    hop_length=320,
    win_length=1024,
    n_mels=64,
)


def compute_log_mel_spectrogram(y: np.ndarray,
                                 sample_rate: int = 16000,
                                 n_fft: int = 1024,
                                 hop_length: int = 320,
                                 win_length: int = 1024,
                                 n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0)
    return log_mel.astype(np.float32)


def compute_low_freq_ratio(y: np.ndarray, sample_rate: int = 16000,
                            cutoff_hz: float = 150.0) -> float:
    """
    Fraction of signal energy below cutoff_hz. Footsteps are dominated
    by a low-frequency thump; speech, claps, and most confusable sounds
    carry much more energy above this cutoff. Used both as an auxiliary
    input to the CNN and as an explicit safety gate at inference time.
    """
    n = len(y)
    if n == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total_energy = np.sum(spectrum ** 2) + 1e-12
    low_energy = np.sum(spectrum[freqs <= cutoff_hz] ** 2)
    return float(low_energy / total_energy)


def spec_augment(log_mel: np.ndarray, n_freq_masks: int = 1, freq_mask_width: int = 8,
                  n_time_masks: int = 1, time_mask_width: int = 8) -> np.ndarray:
    """
    Randomly zero out frequency and time bands. Applied only during
    training (see dataset.py) -- forces the model to not over-rely on
    any single narrow band or time slice, which matters more as the
    negative class grows more acoustically diverse (speech, claps, etc).
    """
    log_mel = log_mel.copy()
    n_mels, n_frames = log_mel.shape

    for _ in range(n_freq_masks):
        if n_mels <= freq_mask_width:
            continue
        f0 = np.random.randint(0, n_mels - freq_mask_width)
        log_mel[f0:f0 + freq_mask_width, :] = 0.0

    for _ in range(n_time_masks):
        if n_frames <= time_mask_width:
            continue
        t0 = np.random.randint(0, n_frames - time_mask_width)
        log_mel[:, t0:t0 + time_mask_width] = 0.0

    return log_mel
