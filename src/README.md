# Fall Detection (PyTorch & Transformer)

This repository contains a modular PyTorch implementation of a Fall Detection system using MediaPipe and YOLOv11 for skeletal keypoint extraction, followed by a Transformer Encoder for sequence classification. It was rewritten from a TensorFlow Keras `.ipynb` file to allow for easy cloning and usage inside Kaggle kernels.

## Setup for Kaggle
In your Kaggle Notebook, run the following:

```bash
# 1. Clone this repository directly
!git clone <YOUR-GITHUB-REPO-URL> fall_detection_pytorch
%cd fall_detection_pytorch

# 2. Install dependencies
!pip install -r requirements.txt

# 3. View the configuration (modify default.yaml if needed for your dataset paths)
# Make sure DATASET_DIR in configs/default.yaml points to your Kaggle input directory!
!cat configs/default.yaml
```

## Workflow

### 1. Data Preparation
The LEI2 dataset contains videos and text files with `fall_start` and `fall_end` frames. To parse these videos, run MediaPipe and YOLO over them, and extract feature sequences (saved as `.npy` arrays), run:

```bash
python prepare_data.py
```
*Note: Make sure your `DATASET_DIR` in `configs/default.yaml` matches the path to your dataset in Kaggle (e.g. `/kaggle/input/falldataset-imvia`).*

### 2. Training
Run the training loop, which trains the PyTorch Transformer model and saves the best checkpoint based on validation loss.

```bash
python train.py
```

### 3. Evaluation
Evaluate the trained model on the held-out test set:

```bash
python evaluate.py
```

### 4. Inference Demo
Run inference on a single video file or webcam stream (0) with smooth rolling window tracking:

```bash
python inference.py --video test_video.mp4 --output result.mp4
```

## Modifying the Model
The core PyTorch model is located at `fall_detection/models/transformer.py`. It uses a fully parameterizable self-attention `TransformerEncoder` acting on 66-dimensional inputs extracted by MediaPipe. You can tweak `num_layers`, `d_model`, etc. directly inside `configs/default.yaml`.
