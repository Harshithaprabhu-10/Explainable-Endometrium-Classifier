# Explainable Histopathology Classifier

A deep-learning ensemble (**ConvNeXt-Tiny + ResNet-50**) that classifies microscopic tissue images into four classes and explains each prediction with **Grad-CAM** heatmaps through a simple web interface.

---

## Overview

Two CNN architectures with different strengths are trained on the same dataset and their softmax outputs are averaged. The ensemble reaches **83.43 % validation accuracy**. To make the model's decision transparent, Grad-CAM is applied to both networks and the result is shown in a Gradio app:

1. Upload a microscopic image.
2. Get the probability of each class.
3. See a heatmap showing which regions drove the decision.
4. Read an automatic, region-by-region explanation.

## Features

- Four-class prediction with percentage scores
- Weighted ensemble of ConvNeXt-Tiny and ResNet-50
- Grad-CAM heatmaps: ensemble, ConvNeXt only, ResNet only (equal-size panels)
- Numbered "hot regions" with location, area, share of attention and which model drives each one
- Plain-language summary: confidence margin, model agreement, attention overlap
- Warning when a highlighted region looks like empty space or background

## Classes

| Label | Meaning |
|-------|---------|
| EA | Endometrial adenocarcinoma |
| EH | Endometrial hyperplasia |
| EP | Endometrial polyp |
| NE | Normal endometrium |

## Method

### Preprocessing
- Images are resized and converted with `ToTensor()` only.
- **No ImageNet mean/std normalization** is used, in training or in inference. Adding it at inference would change the predictions.

### ConvNeXt-Tiny (`train.py`)
- `timm.create_model("convnext_tiny", pretrained=True, num_classes=4)`
- Input 256 × 256, Adam optimizer, learning rate 3e-5, batch size 8, 30 epochs
- Augmentation: `RandomResizedCrop(256, scale=(0.85, 1.0))`, horizontal flip, rotation ±10°, colour jitter (brightness and contrast 0.15)
- Best model (by validation accuracy) is saved to `best_model.pth`; training can resume from `checkpoint.pth`

### ResNet-50
- `torchvision.models.resnet50` with the final layer replaced by `Dropout(0.75)` followed by `Linear(2048, 4)`
- Weights are stored as a checkpoint dictionary under the key `"model"`

### Ensemble (`ensemble.py`)
Softmax probabilities of both models are combined as

```
p_final = w · p_ConvNeXt + (1 − w) · p_ResNet
```

`w` is searched from 0.05 to 0.95 in steps of 0.05 on the validation set. The chosen setting is **w = 0.50** (ConvNeXt 0.50, ResNet 0.50).

### Grad-CAM (`app.py`)
- ConvNeXt target layer: `model.stages[-1].blocks[-1]`
- ResNet target layer: `model.layer4[-1]`
- For the predicted class, each model's gradients are averaged per channel and used to weight that layer's activations. The result is passed through ReLU, upsampled and normalized to [0, 1].
- The ensemble heatmap is the weighted sum of both maps, using the same weights as the prediction.

## Results

Validation accuracy of the ensemble for different ConvNeXt weights (ResNet weight = 1 − w):

| ConvNeXt weight | Accuracy |
|-----------------|----------|
| 0.05 | 54.75 % |
| 0.25 | 59.19 % |
| 0.45 | 73.33 % |
| **0.50** | **83.43 %** |
| 0.55 | 82.83 % |
| 0.60 – 0.65 | 83.03 % |
| 0.70 – 0.80 | 83.43 % |
| 0.85 – 0.95 | 83.23 % |

Accuracy is flat at about 83 % once the ConvNeXt weight is 0.50 or higher, so the result is not sensitive to the exact weight in that range. Give ConvNeXt at least half the weight; below 0.45 accuracy drops sharply.

| Model | Validation accuracy |
|-------|---------------------|
| ConvNeXt-Tiny alone | _add_ |
| ResNet-50 alone | _add_ |
| **Ensemble (0.5 / 0.5)** | **83.43 %** |

See `confusion_matrix.png` and `roc_curve.png` for per-class behaviour.

## Installation

Python 3.9 or newer is recommended.

```bash
# create and activate a virtual environment
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # Linux / macOS

# install dependencies
pip install torch torchvision timm gradio opencv-python numpy pillow
```

## Usage

### 1. Prepare the data

```bash
python split_dataset.py
```

### 2. Train ConvNeXt
```bash
python train.py
```

### 3. Evaluate the ensemble
```bash
python ensemble.py
```

### 4. Launch the interface
```bash
python app.py
```

Open the local link printed in the terminal (usually `http://127.0.0.1:7860`), upload an image and click **Analyze**.

The app expects `convnext_model.pth` and `resnet_model.pth` in the same folder.

## How the explanation works

The interface returns three things for each image:

1. **Class probabilities** from the ensemble.
2. **Summary** covering:
   - confidence: gap between the top two classes
   - agreement: whether both models predict the same class
   - focus: where the heat is concentrated and how much of the image it covers
   - overlap: how much the two models' hot regions coincide (IoU)
3. **Heatmap regions**: the strongest connected areas (heat above 0.5) are numbered and described by position, area, share of total attention, which model drives them, and whether they are darker or paler than the rest of the image.

**How to read the heatmap:** red and yellow mean strong influence on the prediction; blue means little influence.

Grad-CAM shows *where* the network found evidence for a class. It does not state *why* in pathological terms. The "appearance" column only compares brightness with the rest of the image and does not identify glands, nuclei or other structures.


## Author

Harshitha
