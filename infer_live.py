"""
infer_live.py

Real-time footstep detection from a microphone -- built to actually
work with external USB mics/interfaces, not just a laptop's built-in
mic. The earlier version hardcoded channels=1 and dtype="float32" when
opening the stream, which many external devices reject outright even
when the sample rate is fine (that's why only the laptop mic worked).

This version probes real combinations of sample rate, channel count,
and dtype against the chosen device until one actually opens, then
downmixes/resamples/converts every incoming chunk to mono float32 at
the model's target rate before it reaches the ring buffer -- so nothing
downstream needs to know or care what the device natively supports.

Usage:
    python infer_live.py --checkpoint checkpoints/best_model.pt
    python infer_live.py --checkpoint checkpoints/best_model.pt --device_name "USB"
    python infer_live.py --checkpoint checkpoints/best_model.pt --list_devices
"""

import argparse
import time
import queue
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch
import librosa

from preprocessing.audio import preprocess_buffer
from preprocessing.features import compute_log_mel_spectrogram, compute_low_freq_ratio
from models.cnn import FootstepCNN

try:
    import sounddevice as sd
except OSError:
    print("ERROR: could not load PortAudio / sounddevice. On Linux install "
          "with: sudo apt install portaudio19-dev  then: pip install sounddevice")
    raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--infer_interval_sec", type=float, default=0.3)
    p.add_argument("--prob_threshold", type=float, default=0.80)
    p.add_argument("--min_low_freq_ratio", type=float, default=0.35,
                    help="minimum low-frequency energy fraction required to accept "
                         "a footstep prediction -- rejects speech/claps/etc even if "
                         "the CNN is fooled")
    p.add_argument("--smoothing_window", type=int, default=3)
    p.add_argument("--smoothing_min_hits", type=int, default=2)
    p.add_argument("--input_block_sec", type=float, default=0.05)
    p.add_argument("--device_index", type=int, default=None,
                    help="sounddevice input device index; default = system default mic")
    p.add_argument("--device_name", type=str, default=None,
                    help="substring to match against device names instead of an index "
                         "-- more robust than --device_index since device indices shift "
                         "whenever hardware is plugged/unplugged")
    p.add_argument("--list_devices", action="store_true",
                    help="print all input devices with their supported rates and exit")
    return p.parse_args()


class RingBuffer:
    def __init__(self, size_samples: int):
        self.size = size_samples
        self.buf = np.zeros(size_samples, dtype=np.float32)

    def push(self, chunk: np.ndarray):
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            return
        self.buf = np.roll(self.buf, -n)
        self.buf[-n:] = chunk

    def snapshot(self) -> np.ndarray:
        return self.buf.copy()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    use_aux = ckpt["model_arch"].get("use_aux_features", False)
    dropout = ckpt["model_arch"].get("dropout", 0.4)
    model = FootstepCNN(n_mels=ckpt["model_arch"]["n_mels"], dropout=dropout,
                         use_aux_features=use_aux).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def resolve_device_index(args) -> int:
    if args.device_name is not None:
        matches = []
        for idx, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0 and args.device_name.lower() in info["name"].lower():
                matches.append((idx, info["name"]))
        if not matches:
            print(f"ERROR: no input device matched name '{args.device_name}'")
            print("Available input devices:")
            print(sd.query_devices())
            sys.exit(1)
        print(f"Matched device_name '{args.device_name}' -> index {matches[0][0]} ({matches[0][1]})")
        if len(matches) > 1:
            print(f"  (multiple matches: {[m[1] for m in matches]}, using the first)")
        return matches[0][0]
    return args.device_index


def find_working_stream_config(device_index, target_sr: int):
    """
    Try real combinations of sample rate, channel count, and dtype
    against the device until sd.check_input_settings() accepts one.
    This is the actual fix for external mics: a hardcoded single
    (samplerate, channels=1, dtype=float32) guess fails silently on
    many devices that would otherwise work fine with a different
    combination.
    """
    try:
        info = sd.query_devices(device_index, kind="input")
        native_sr = int(info["default_samplerate"])
        max_channels = max(1, int(info["max_input_channels"]))
    except Exception:
        native_sr = target_sr
        max_channels = 1

    candidate_rates = list(dict.fromkeys([native_sr, target_sr, 44100, 48000, 16000]))
    candidate_channels = list(dict.fromkeys([1, max_channels] if max_channels >= 1 else [1]))
    candidate_dtypes = ["float32", "int16"]

    for sr in candidate_rates:
        for ch in candidate_channels:
            for dtype in candidate_dtypes:
                try:
                    sd.check_input_settings(device=device_index, samplerate=sr,
                                             channels=ch, dtype=dtype)
                    return sr, ch, dtype
                except Exception:
                    continue
    return None


def main():
    args = parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.checkpoint} ...")
    model, ckpt = load_model(args.checkpoint, device)
    sr = ckpt["sample_rate"]
    duration_sec = ckpt["duration_sec"]
    mel_params = ckpt["mel_params"]
    print(f"Loaded model. sample_rate={sr}, duration_sec={duration_sec}, mel_params={mel_params}")

    device_index = resolve_device_index(args)

    config = find_working_stream_config(device_index, sr)
    if config is None:
        print(f"ERROR: could not find a working (samplerate, channels, dtype) combination "
              f"for device_index={device_index}.")
        print("Available input devices:")
        print(sd.query_devices())
        return
    capture_sr, capture_channels, capture_dtype = config
    print(f"Opening stream: samplerate={capture_sr}, channels={capture_channels}, "
          f"dtype={capture_dtype} (will convert to mono {sr}Hz float32 internally)")

    ring = RingBuffer(size_samples=int(duration_sec * sr))
    recent_predictions = deque(maxlen=args.smoothing_window)
    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[mic warning] {status}", file=sys.stderr)
        chunk = indata.astype(np.float32)
        if capture_dtype == "int16":
            chunk = chunk / 32768.0
        if chunk.ndim > 1 and chunk.shape[1] > 1:
            chunk = chunk.mean(axis=1)  # downmix stereo/multi-channel to mono
        else:
            chunk = chunk.reshape(-1)
        if capture_sr != sr:
            chunk = librosa.resample(chunk, orig_sr=capture_sr, target_sr=sr)
        audio_q.put(chunk)

    block_size = int(args.input_block_sec * capture_sr)

    try:
        stream = sd.InputStream(
            samplerate=capture_sr,
            channels=capture_channels,
            dtype=capture_dtype,
            blocksize=block_size,
            device=device_index,
            callback=audio_callback,
        )
    except Exception as e:
        print(f"ERROR: could not open microphone stream even after probing: {e}")
        print("Available input devices:")
        print(sd.query_devices())
        return

    print("\nListening for footsteps... (Ctrl+C to stop)\n")
    last_infer_time = time.time()

    try:
        with stream:
            while True:
                try:
                    while True:
                        chunk = audio_q.get_nowait()
                        ring.push(chunk)
                except queue.Empty:
                    pass

                now = time.time()
                if now - last_infer_time >= args.infer_interval_sec:
                    last_infer_time = now

                    raw = ring.snapshot()
                    processed = preprocess_buffer(raw, sr=sr, duration_sec=duration_sec)
                    log_mel = compute_log_mel_spectrogram(
                        processed,
                        sample_rate=mel_params["sample_rate"],
                        n_fft=mel_params["n_fft"],
                        hop_length=mel_params["hop_length"],
                        win_length=mel_params["win_length"],
                        n_mels=mel_params["n_mels"],
                    )
                    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)
                    low_freq_ratio = compute_low_freq_ratio(processed, sample_rate=sr)
                    aux_tensor = torch.tensor([[low_freq_ratio]], dtype=torch.float32).to(device)

                    with torch.no_grad():
                        logits = model(tensor, aux_tensor if model.use_aux_features else None)
                        probs = torch.softmax(logits, dim=1)[0]
                        footstep_prob = probs[1].item()

                    passes_freq_gate = low_freq_ratio >= args.min_low_freq_ratio
                    single_hit = (footstep_prob >= args.prob_threshold) and passes_freq_gate
                    recent_predictions.append(single_hit)
                    hits = sum(recent_predictions)
                    is_footstep = (len(recent_predictions) == args.smoothing_window and
                                    hits >= args.smoothing_min_hits)

                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    label = "FOOTSTEP DETECTED" if is_footstep else "NO FOOTSTEP"
                    conf = footstep_prob if is_footstep else (1 - footstep_prob)
                    print(f"[{ts}] {label} | confidence={conf*100:.1f}% | low_freq_ratio={low_freq_ratio:.2f}")

                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
