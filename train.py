import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy
import time

# ==========================================
# 1. SETUP
# ==========================================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0005
IMG_SIZE = 224

# ==========================================
# 2. DATA AUGMENTATION
# ==========================================
train_transforms = transforms.Compose(
    [
        # 1. Geometric & Color Transforms (Work on PIL Images)
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        # 2. Convert to Tensor (Crucial Step!)
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(root="images/TRAIN", transform=train_transforms)
test_dataset = datasets.ImageFolder(root="images/TEST", transform=test_transforms)
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

print(f"Classes: {class_names}")
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

# ==========================================
# 3. RESNET18 MODEL
# ==========================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# FREEZE ALL LAYERS except the final 30 layers
for param in list(model.parameters())[:-30]:
    param.requires_grad = False

# Only train the final 30 layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.6),  # Increased dropout
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),  # Add batch norm
    nn.Dropout(0.4),
    nn.Linear(128, NUM_CLASSES),
)

model = model.to(DEVICE)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(
    f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)"
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


# ==========================================
# 4. TRAINING WITH EARLY STOPPING
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 5
    no_improve_count = 0

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        # Gradient clipping to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "test":
                scheduler.step(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improve_count = 0
                    print(f"✓ New best accuracy: {best_acc:.4f}")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": best_model_wts,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_acc": best_acc,
                            "class_names": class_names,
                            "num_classes": NUM_CLASSES,
                            "img_size": IMG_SIZE,
                        },
                        "best_model.pth",
                    )
                    print(f"💾 Best model saved!")
                else:
                    no_improve_count += 1
                    print(f"No improvement. Patience: {no_improve_count}/{patience}")

        if no_improve_count >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


# Run Training
trained_model, history = train_model(
    model, criterion, optimizer, scheduler, num_epochs=EPOCHS
)

# ==========================================
# 5. PLOT & EVALUATE
# ==========================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["test_acc"], label="Test Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["test_loss"], label="Test Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Final Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np


def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


evaluate_model(trained_model, test_loader, class_names)
