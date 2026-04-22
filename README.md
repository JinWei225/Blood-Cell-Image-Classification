# 🩸 Blood Cell Image Classification (PyTorch)

A deep learning project using **PyTorch** and **Transfer Learning (ResNet18)** to classify blood cell images into four categories: **Eosinophil**, **Lymphocyte**, **Monocyte**, and **Neutrophil**.

This project demonstrates data preprocessing, model training with early stopping, evaluation metrics, and inference on Apple Silicon (MPS).

## 📊 Project Overview

- **Task**: Multi-class Image Classification
- **Dataset**: ~12,000 Blood Cell Images (Train + Test)
- **Classes**: 4 (Eosinophil, Lymphocyte, Monocyte, Neutrophil)
- **Model Architecture**: ResNet18 (Pre-trained on ImageNet)
- **Framework**: PyTorch & Torchvision
- **Hardware**: Optimized for Apple Silicon (M1/M2/M3 MPS)

## 🚀 Features

- ✅ **Transfer Learning**: Utilizes pre-trained ResNet18 weights for faster convergence.
- ✅ **Data Augmentation**: Random flips, rotation, and color jittering to prevent overfitting.
- ✅ **Early Stopping**: Automatically stops training when validation accuracy plateaus.
- ✅ **MPS Support**: Fully compatible with Mac Mini/MacBook Pro GPU acceleration.
- ✅ **Evaluation**: Generates Classification Reports and Confusion Matrices.
- ✅ **Inference**: Easy-to-use script for predicting new images.

## 📁 Directory Structure

```text
Blood-Cell-Image-Classification/
├── images/
│   ├── TRAIN/
│   │   ├── EOSINOPHIL/
│   │   ├── LYMPHOCYTE/
│   │   ├── MONOCYTE/
│   │   └── NEUTROPHIL/
│   └── TEST/
│       ├── EOSINOPHIL/
│       ├── ...
├── train.py                # Main training script
├── predict.py              # Inference script
├── best_model.pth          # Saved best model weights (generated after training)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Blood-Cell-Image-Classification.git
cd Blood-Cell-Image-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` content:**

```text
torch>=2.1.0
torchvision>=0.16.0
matplotlib
numpy
scikit-learn
seaborn
pillow
```

> **Note for Mac Users:** Ensure you have the latest version of PyTorch installed to support MPS (Metal Performance Shaders):
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> # Or check pytorch.org for the latest MPS-compatible command
> ```

## 🏋️‍♂️ Training

To train the model, simply run:

```bash
python train.py
```

### Training Details

- **Optimizer**: SGD with Momentum (`lr=0.001`, `momentum=0.9`)
- **Scheduler**: ReduceLROnPlateau (reduces LR if val acc doesn't improve)
- **Batch Size**: 32
- **Early Stopping**: Patience of 15 epochs
- **Augmentation**: Horizontal/Vertical Flip, Rotation (±10°), Color Jitter

The script will automatically save the best model as `best_model.pth` whenever the test accuracy improves.

## 🔮 Inference (Prediction)

To predict the class of a single blood cell image:

1. Open `predict.py` and update the `image_path` variable.
2. Run the script:

```bash
python predict.py
```

**Example Output:**

```text
✓ Model loaded successfully!
Classes: ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
Best accuracy: 0.8524

Prediction: NEUTROPHIL
Confidence: 92.45%
```

## 📈 Results

The final model achieved approximately **85% accuracy** on the test set.

| Metric                | Value                       |
| :-------------------- | :-------------------------- |
| **Test Accuracy**     | ~85.2%                      |
| **Training Accuracy** | ~95.0%                      |
| **Model Size**        | ~45 MB (ResNet18)           |
| **Inference Time**    | < 50ms per image (on M1/M2) |

_(Add a screenshot of your confusion matrix or loss curve here if desired)_

## 💻 Hardware Support

This code is optimized for **Apple Silicon (MPS)**. If you are running on NVIDIA GPU or CPU, the code will automatically detect the device:

```python
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Change to "cuda" if you have an NVIDIA GPU
```

## 🙏 Acknowledgments

- Dataset provided by [Source of Dataset, e.g., Kaggle/UCI]
- PyTorch Documentation
- Hugging Face Transformers & Datasets libraries

---
