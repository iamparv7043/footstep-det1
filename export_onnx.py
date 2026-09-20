"""
export_onnx.py

Exports a trained checkpoint (checkpoints/best_model.pt) to ONNX format
for edge deployment. ONNX Runtime runs on Raspberry Pi, Jetson, Linux
SBCs, and Windows/Mac -- without needing PyTorch installed on the
target device, and typically with lower inference latency than
PyTorch's own eager-mode execution.

The exported .onnx file is self-describing (input/output shapes are
fixed at export time), but does NOT carry the preprocessing parameters
(sample rate, mel params, duration) the way the .pt checkpoint does --
those still need to travel with your deployment code separately. This
script also writes a small deployment_config.json alongside the .onnx
file with exactly those parameters, so nothing has to be hardcoded or
guessed on the target device.

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/model.onnx
"""

import argparse
import json

import torch
import numpy as np

from models.cnn import FootstepCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--output", type=str, default="checkpoints/model.onnx")
    p.add_argument("--opset", type=int, default=13,
                    help="ONNX opset version -- 13 is broadly compatible with "
                         "ONNX Runtime versions available on most edge boards")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")  # export from CPU for maximum portability

    ckpt = torch.load(args.checkpoint, map_location=device)
    use_aux = ckpt["model_arch"].get("use_aux_features", False)
    dropout = ckpt["model_arch"].get("dropout", 0.4)
    n_mels = ckpt["model_arch"]["n_mels"]

    model = FootstepCNN(n_mels=n_mels, dropout=dropout, use_aux_features=use_aux)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Compute expected number of time frames from the checkpoint's own
    # mel params, so the dummy input shape matches real inference exactly.
    mel_params = ckpt["mel_params"]
    sr = ckpt["sample_rate"]
    duration_sec = ckpt["duration_sec"]
    n_samples = int(duration_sec * sr)
    n_frames = 1 + n_samples // mel_params["hop_length"]

    dummy_x = torch.randn(1, 1, n_mels, n_frames)
    dummy_aux = torch.randn(1, 1) if use_aux else None

    input_names = ["log_mel_spectrogram"]
    output_names = ["logits"]
    dynamic_axes = {"log_mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}}

    if use_aux:
        input_names.append("low_freq_ratio")
        dynamic_axes["low_freq_ratio"] = {0: "batch"}
        export_args = (dummy_x, dummy_aux)
    else:
        export_args = (dummy_x,)

    torch.onnx.export(
        model, export_args, args.output,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # use the classic TorchScript-based exporter -- more
                        # stable for CNNs and doesn't require the extra
                        # onnxscript dependency the newer exporter needs
    )
    print(f"Exported ONNX model to {args.output}")

    # Write the preprocessing/deployment config alongside the model --
    # the target device needs these exact values to preprocess audio
    # identically to training.
    config = {
        "sample_rate": sr,
        "duration_sec": duration_sec,
        "mel_params": mel_params,
        "n_frames_expected": n_frames,
        "use_aux_features": use_aux,
        "class_names": ckpt["class_names"],
        "input_names": input_names,
        "output_names": output_names,
        "note": "logits are raw scores for [no_footstep, footstep]; "
                "apply softmax on-device to get probabilities.",
    }
    config_path = args.output.rsplit(".", 1)[0] + "_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote deployment config to {config_path}")

    # --- Verify the exported model actually loads and runs correctly ---
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime not installed -- skipping verification.")
        print("Install with: pip install onnxruntime")
        return

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    ort_inputs = {"log_mel_spectrogram": dummy_x.numpy().astype(np.float32)}
    if use_aux:
        ort_inputs["low_freq_ratio"] = dummy_aux.numpy().astype(np.float32)

    with torch.no_grad():
        torch_out = model(*export_args).numpy()
    ort_out = sess.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"\nVerification: PyTorch vs ONNX Runtime max output difference = {max_diff:.2e}")
    if max_diff < 1e-4:
        print("PASS -- ONNX export matches PyTorch model output.")
    else:
        print("WARNING -- outputs differ more than expected, inspect before deploying.")


if __name__ == "__main__":
    main()
