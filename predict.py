import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================
# Change to "cuda" if you are using NVIDIA GPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load("best_model.pth", map_location=DEVICE, weights_only=True)

# Recreate model architecture
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(num_ftrs, checkpoint["num_classes"])
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Get class names
class_names = checkpoint["class_names"]
img_size = checkpoint["img_size"]

print(f"✓ Model loaded successfully!")
print(f"Classes: {class_names}")
print(f"Best accuracy: {checkpoint['best_acc']:.4f}")


# ==========================================
# 2. PREDICTION FUNCTION
# ==========================================
def predict_image(image_path):
    """Predict class for a single image"""

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    # Load and process image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    predicted_class = class_names[preds.item()]
    confidence = probs[0][preds.item()].item()

    return predicted_class, confidence, probs


# ==========================================
# 3. TEST PREDICTION
# ==========================================
# Example: Predict on a test image
image_path = "images/TEST/EOSINOPHIL/some_image.jpeg"  # Replace with your image
predicted_class, confidence, all_probs = predict_image(image_path)

print(f"\nPrediction: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print("\nAll class probabilities:")
for cls, prob in zip(class_names, all_probs[0].cpu().numpy()):
    print(f"  {cls}: {prob:.2%}")
