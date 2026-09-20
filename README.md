\# Real-Time Footstep Detection



A software-based real-time human footstep detection system using microphone audio and a lightweight CNN model.



The system processes incoming audio, extracts time-frequency features, classifies short audio segments as \*\*footstep\*\* or \*\*no footstep\*\*, and performs continuous prediction for real-time detection.



\## Features



\* Real-time microphone-based inference

\* CNN-based audio classification

\* Log-Mel spectrogram feature extraction

\* Audio preprocessing and segmentation

\* Training, evaluation, and live inference scripts

\* ONNX model export support

\* Dataset organized into `footstep` and `no\_footstep` classes

\* Lightweight model suitable for real-time applications



\## Project Structure



```text

footstep\_detection/

│

├── dataset/

│   ├── footstep/

│   └── no\_footstep/

│

├── models/

│   ├── \_\_init\_\_.py

│   └── cnn.py

│

├── preprocessing/

│   ├── \_\_init\_\_.py

│   ├── audio.py

│   └── features.py

│

├── checkpoints/

│

├── dataset.py

├── train.py

├── evaluate.py

├── infer\_live.py

├── export\_onnx.py

├── requirements.txt

├── .gitignore

└── README.md

```



\## How It Works



The processing pipeline is:



```text

Microphone

&#x20;   ↓

Audio Capture

&#x20;   ↓

Preprocessing

&#x20;   ↓

Feature Extraction

&#x20;   ↓

Log-Mel Spectrogram

&#x20;   ↓

CNN Classifier

&#x20;   ↓

Temporal Prediction / Smoothing

&#x20;   ↓

Footstep / No Footstep

```



\## Requirements



\* Python 3.10+ recommended

\* PyTorch

\* NumPy

\* SciPy

\* Librosa

\* SoundDevice

\* SoundFile

\* Scikit-learn

\* Matplotlib



For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver.



\## Installation



Clone the repository:



```bash

git clone https://github.com/iamparv7043/footstep-det1.git

cd footstep-det1

```



Create a virtual environment:



\### Windows



```powershell

python -m venv venv

```



Activate it:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



If PowerShell blocks script execution, run:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

```



and activate again:



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### Linux



```bash

python3 -m venv venv

source venv/bin/activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Dataset



The dataset should contain two classes:



```text

dataset/

├── footstep/

│   ├── footstep\_001.wav

│   ├── footstep\_002.wav

│   └── ...

│

└── no\_footstep/

&#x20;   ├── no\_footstep\_001.wav

&#x20;   ├── no\_footstep\_002.wav

&#x20;   └── ...

```



Place your WAV files in the appropriate directory.



The `no\_footstep` class should contain sounds that the model may encounter during normal operation, such as:



\* Speech

\* Background noise

\* Room ambience

\* Non-footstep sounds

\* Other sounds that could cause false positives



This helps the model learn to distinguish footsteps from other acoustic events.



\## Training



After preparing the dataset, train the CNN model using:



```bash

python train.py

```



The training script processes the dataset, creates training/validation/test splits, trains the model, and saves the trained checkpoint.



The trained model is stored in the `checkpoints/` directory.



\## Evaluation



After training, evaluate the trained model using:



```bash

python evaluate.py

```



This evaluates the model on the test data and reports classification performance.



Typical metrics include:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Confusion matrix



\## Real-Time Inference



Connect a microphone to your computer and run:



```bash

python infer\_live.py

```



The application continuously captures microphone audio and performs predictions in real time.



The output indicates whether the current audio segment is classified as:



```text

FOOTSTEP

```



or



```text

NO FOOTSTEP

```



Make sure the required trained checkpoint exists in the expected `checkpoints/` location before starting live inference.



\## ONNX Export



The trained PyTorch model can be exported to ONNX using:



```bash

python export\_onnx.py

```



The ONNX model can then be used with ONNX Runtime or integrated into other applications.



\## GPU Support



To check whether PyTorch detects your NVIDIA GPU:



```bash

python -c "import torch; print('CUDA:', torch.cuda.is\_available()); print('GPU:', torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')"

```



Example output:



```text

CUDA: True

GPU: NVIDIA GeForce RTX 4050 Laptop GPU

```



If CUDA is unavailable, the project can still run using the CPU, although training may take longer.



\## Model



The classifier is implemented in:



```text

models/cnn.py

```



The model receives extracted audio features and predicts one of two classes:



```text

0 → No Footstep

1 → Footstep

```



\## Preprocessing



Audio processing utilities are implemented in:



```text

preprocessing/audio.py

preprocessing/features.py

```



These modules handle audio loading, preprocessing, segmentation, and feature extraction.



\## Important Note About Dataset



The original audio dataset is not included in this repository.



WAV files are excluded using `.gitignore` to avoid unnecessarily large repository size and because dataset redistribution may depend on the source and its licensing terms.



You must provide your own dataset before training.



\## Limitations



This is a software prototype for real-time footstep detection.



Performance can depend on:



\* Microphone quality

\* Recording environment

\* Distance from the microphone

\* Background noise

\* Footstep type

\* Floor and surface characteristics

\* Dataset size and diversity



False positives can occur when acoustic events resemble footsteps. Including representative non-footstep sounds in the training dataset can help improve robustness.



\## Future Improvements



\* Larger and more diverse datasets

\* Better background-noise handling

\* Improved false-positive rejection

\* Model quantization

\* ONNX Runtime deployment

\* Mobile deployment

\* Edge-device deployment

\* Multi-class acoustic event detection

\* FPGA/embedded implementation



\## License



Add an appropriate license before redistributing the project or dataset.



\## Author



\*\*Parv Voraliya\*\*



GitHub: https://github.com/iamparv7043



