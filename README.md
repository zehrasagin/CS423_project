# Emotion Recognition from Facial Expressions

**Authors**

- Zehra Sağın
- İrem Akova
- Beril Eda Teberci

## Overview

This project performs **facial emotion recognition** from webcam input using deep learning and real-time face detection. It includes:

- a **MobileNetV2-based transfer learning model**
- a **custom CNN model**
- a **Tkinter GUI** for live use
- standalone **real-time inference scripts**
- a simple **face detection demo**

The system predicts one of the following seven emotions:

- Surprise
- Fear
- Disgust
- Happy
- Sad
- Angry
- Neutral

The project is designed so that a reader can both:

- understand the model pipeline and code organization
- run the training and inference scripts with minimal setup

## Project Goals

- Train emotion recognition models on the **RAF-DB** dataset
- Compare a transfer learning baseline with a custom CNN
- Run live webcam-based emotion recognition
- Apply visual effects to the video feed based on the predicted emotion
- Provide both a command-line workflow and a GUI workflow

## Repository Structure

```text
CS423_project/
├── data/
│   └── RAF_DB/
│       └── archive-3/
│           └── DATASET/
│               ├── train/
│               └── test/
├── models/
│   ├── emotion_custom_cnn.keras
│   └── emotion_mobilenetv2.keras
├── emotion_gui.py
├── face_detection.py
├── real_time_custom_cnn.py
├── real_time_emotion_filter.py
├── requirements.txt
├── train_custom_cnn.py
└── train_model.py
```

## Main Files

### `emotion_gui.py`

Tkinter-based graphical interface for live webcam emotion recognition.

- lets the user choose between **MobileNetV2** and **Custom CNN**
- starts and stops the webcam
- shows predicted emotion, confidence, and FPS
- optionally applies an emotion-based visual filter to the live video

### `train_custom_cnn.py`

Trains the custom CNN model on RAF-DB.

- uses `image_dataset_from_directory`
- performs data augmentation
- saves the best model to `models/emotion_custom_cnn.keras`
- prints a classification report and confusion matrix after evaluation

### `train_model.py`

Trains the MobileNetV2-based model.

- uses transfer learning from ImageNet
- trains a classifier head first
- fine-tunes the last MobileNetV2 layers
- saves the best model to `models/emotion_mobilenetv2.keras`

### `real_time_custom_cnn.py`

Runs the custom CNN model in real time from the webcam.

- detects the face using MediaPipe
- aligns the face
- enhances contrast using CLAHE
- applies simple test-time augmentation
- smooths predictions over recent frames
- overlays an emotion-specific visual filter on the displayed frame

### `real_time_emotion_filter.py`

Legacy real-time webcam script for the MobileNetV2 model.

- performs real-time emotion recognition
- applies emotion-based visual effects
- useful as a direct non-GUI demo

### `face_detection.py`

Simple MediaPipe face detection demo.

- detects faces
- draws bounding boxes
- shows cropped face output
- useful for checking that the webcam and face detector are working

## Dataset

This project expects the **RAF-DB** dataset to already be extracted into the following directory:

```text
data/RAF_DB/archive-3/DATASET/
```

Inside that folder, the structure should be:

```text
DATASET/
├── train/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
│   ├── 6/
│   └── 7/
└── test/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    └── 7/
```

### Label Mapping

The numeric RAF-DB folder labels are interpreted as:

| Folder | Emotion |
|---|---|
| `1` | Surprise |
| `2` | Fear |
| `3` | Disgust |
| `4` | Happy |
| `5` | Sad |
| `6` | Angry |
| `7` | Neutral |

If this dataset path is missing, the training scripts will fail. Live inference can still run if the saved model files already exist.

## Environment Setup

### 1. Change into the project directory

```bash
cd /path/to/CS423_project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Important note about TensorFlow

This repository was prepared on **macOS / Apple Silicon**, and `requirements.txt` includes `tensorflow-macos`.

If you are using another operating system and installation fails:

- keep the same project code
- install a TensorFlow 2.15-compatible environment for your platform
- if necessary, remove or replace `tensorflow-macos` during installation

### Optional shortcut

If the `.venv` folder already exists in the repository, you can run scripts directly without activating it:

```bash
./.venv/bin/python emotion_gui.py
```

The README commands below assume the environment is activated and `python` refers to that virtual environment.

## Quick Start

If the models are already present in `models/`, the fastest way to demo the project is:

```bash
python emotion_gui.py
```

Then:

1. Select a model from the dropdown.
2. Click **Start Camera**.
3. Optionally leave **Apply emotion filter** enabled.
4. Click **Stop Camera** or close the window to exit.

## How to Run the Project

### 1. Run the GUI

```bash
python emotion_gui.py
```

What it does:

- opens a Tkinter window
- lets you choose **MobileNetV2** or **Custom CNN**
- loads the selected model automatically
- shows webcam video, emotion label, confidence, and FPS

Notes:

- If the selected model file does not exist, the GUI will show an error.
- For `Custom CNN`, train the model first if `models/emotion_custom_cnn.keras` is missing.
- For `MobileNetV2`, train the model first if `models/emotion_mobilenetv2.keras` is missing.

### 2. Train the Custom CNN

```bash
python train_custom_cnn.py
```

Default behavior:

- dataset directory: `data/RAF_DB/archive-3/DATASET`
- image size: `128`
- batch size: `32`
- epochs: `20`
- saved model path: `models/emotion_custom_cnn.keras`

Useful optional arguments:

```bash
python train_custom_cnn.py --epochs 30
python train_custom_cnn.py --image-size 128 --batch-size 32
python train_custom_cnn.py --dataset-dir data/RAF_DB/archive-3/DATASET
python train_custom_cnn.py --model-path models/emotion_custom_cnn.keras
```

Outputs:

- best model saved to `models/emotion_custom_cnn.keras`
- printed classification report
- printed confusion matrix

### 3. Train the MobileNetV2 Model

```bash
python train_model.py
```

This script:

- loads MobileNetV2 with ImageNet weights
- trains a new classifier head
- fine-tunes the last layers
- saves the best model to `models/emotion_mobilenetv2.keras`

Important notes:

- `train_model.py` uses constants inside the file rather than command-line arguments.
- On the first run, TensorFlow may need to download the pretrained MobileNetV2 weights if they are not already cached.

Outputs:

- best model saved to `models/emotion_mobilenetv2.keras`
- printed classification report
- printed confusion matrix

### 4. Run Real-Time Custom CNN Inference

```bash
python real_time_custom_cnn.py
```

Optional flags:

```bash
python real_time_custom_cnn.py --debug
python real_time_custom_cnn.py --camera-index 0
python real_time_custom_cnn.py --model-path models/emotion_custom_cnn.keras
```

What it does:

- opens the webcam
- detects the most confident face
- crops a square region around the face
- aligns the face using eye landmarks
- enhances the face with CLAHE
- predicts the emotion using the custom CNN
- averages probabilities over recent frames for smoother output
- overlays an emotion filter on the full frame

How to exit:

- click the OpenCV window
- press `q`

### 5. Run Real-Time MobileNetV2 Inference

```bash
python real_time_emotion_filter.py
```

This is the older non-GUI real-time script for the MobileNetV2 model.

How to exit:

- click the OpenCV window
- press `q`

### 6. Run Face Detection Only

```bash
python face_detection.py
```

Use this if you want to test:

- webcam access
- MediaPipe face detection
- face cropping behavior

## Model Pipelines

### Custom CNN Training Pipeline

The custom CNN training script follows this flow:

1. Load RAF-DB images from the `train/` and `test/` directories.
2. Resize each image to `128 x 128`.
3. Apply data augmentation:
   - horizontal flip
   - small random rotation
   - small random zoom
4. Normalize pixel values using `Rescaling(1.0 / 255.0)`.
5. Pass the image through several convolution blocks:
   - `Conv2D(32)`, `Conv2D(32)`
   - `Conv2D(64)`, `Conv2D(64)`
   - `Conv2D(128)`, `Conv2D(128)`
   - `Conv2D(256)`
6. Use `BatchNormalization`, `MaxPooling2D`, and `Dropout` for stability and regularization.
7. Use `GlobalAveragePooling2D` and dense layers for final classification.
8. Output one of seven emotion classes with `softmax`.

### MobileNetV2 Training Pipeline

The MobileNetV2 pipeline follows transfer learning:

1. Load RAF-DB images with `ImageDataGenerator`.
2. Preprocess inputs with MobileNetV2's `preprocess_input`.
3. Load MobileNetV2 without its original top classifier.
4. Freeze the base network at first.
5. Train a new classifier head on top of the extracted features.
6. Unfreeze the last layers and fine-tune them with a smaller learning rate.

### Real-Time Inference Pipeline

#### GUI Pipeline

The GUI uses the following workflow:

1. Open webcam
2. Detect face using MediaPipe
3. Build a square face crop with margin
4. Resize and preprocess input according to the selected model
5. Predict probabilities
6. Smooth predictions over recent frames
7. Show emotion label, confidence, and FPS
8. Optionally apply an emotion-specific full-frame visual filter

#### `real_time_custom_cnn.py` Pipeline

This script is slightly more advanced than the GUI for the custom CNN path:

1. Detect face
2. Crop a square face region
3. Align the face using face mesh eye landmarks
4. Enhance contrast using CLAHE
5. Run simple test-time augmentation:
   - original face
   - horizontally flipped face
6. Average the predictions
7. Smooth probabilities across frames
8. Apply visual overlay effects

## Emotion-Based Visual Filters

These visual filters are applied to the displayed webcam frame:

- **Happy**: increases brightness and saturation
- **Sad**: converts to grayscale and adds blur
- **Angry**: adds a red overlay and stronger contrast
- **Surprise**: overlays detected edges
- **Fear**: adds a dark vignette effect
- **Disgust**: reduces saturation and adds a green tint
- **Neutral / Uncertain / No face**: no filter

## Saved Models

Training produces the following files:

- `models/emotion_custom_cnn.keras`
- `models/emotion_mobilenetv2.keras`

These files are intentionally not committed in Git by default because model binaries are large and are listed in `.gitignore`.

That means after a fresh clone:

- the code may be present
- the dataset may need to be added manually
- the models may need to be retrained

## Current Local Custom CNN Result

For the currently saved `emotion_custom_cnn.keras` model in this workspace, a local evaluation on the RAF-DB test set produced approximately:

| Metric | Value |
|---|---:|
| Accuracy | 0.769 |
| Macro F1 | 0.652 |
| Weighted F1 | 0.764 |

Per-class highlights:

- strongest class: **Happy**
- weaker classes: **Fear** and **Disgust**

These values may change if the model is retrained.

## Suggested Reading and Demo Order

If a teaching assistant or reviewer wants to understand the project quickly, this is a good order:

1. Read this `README.md`.
2. Open `emotion_gui.py` to understand the user-facing application.
3. Open `train_custom_cnn.py` to understand the custom CNN architecture.
4. Open `train_model.py` to understand the MobileNetV2 baseline.
5. Run `python emotion_gui.py` for the fastest demo.
6. Run `python real_time_custom_cnn.py` to see the stronger custom-CNN live pipeline.
7. If needed, retrain a model and inspect the printed classification report and confusion matrix.

## Troubleshooting

### `Model not found`

Reason:

- the corresponding `.keras` file is missing from `models/`

Fix:

- run `python train_custom_cnn.py` for the custom model
- run `python train_model.py` for the MobileNetV2 model

### `Train directory not found` or `Test directory not found`

Reason:

- RAF-DB is not located at the expected path

Fix:

- place the dataset under `data/RAF_DB/archive-3/DATASET/`
- or pass a custom dataset path to `train_custom_cnn.py`

### Webcam could not be opened

Reason:

- camera permissions are blocked
- another application is already using the webcam
- the camera index is wrong

Fix:

- close other webcam applications
- allow camera permissions for Terminal or your Python environment
- try `--camera-index 1` if another index is needed

### Tkinter GUI does not open

Reason:

- Python may not include Tk support in your environment

Fix:

- install a Python distribution with Tkinter support
- on Linux, install the system Tk package such as `python3-tk`

### Dependency installation fails

Reason:

- TensorFlow packages can be platform-specific

Fix:

- use a Python version compatible with TensorFlow 2.15
- if needed, adapt the TensorFlow package selection for your OS

## Notes for Report Submission

This README is intended to complement the written report by making the codebase easy to navigate and reproduce. A reviewer should be able to:

- identify the purpose of each script
- understand which model is used where
- train the models
- run the GUI
- run the standalone real-time demos
- verify outputs such as saved models and classification reports

## License / Academic Use

This repository was prepared as a course project. If you share or reuse it, make sure the RAF-DB dataset is used according to its original license and access conditions.
