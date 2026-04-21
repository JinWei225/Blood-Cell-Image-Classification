import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

# 1. Setup Data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(root="images/TRAIN", transform=transform)
simple_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

data_iter = iter(simple_dataloader)
images, labels = next(data_iter)

# 2. Corrected Plotting Logic
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 cols

# Flatten the axes array so we can iterate through it easily (shape becomes [10])
axes_flat = axes.flatten()

for i in range(10):
    # Get the image and label
    img_tensor = images[i]
    label_idx = labels[i].item()  # .item() converts tensor int to python int
    class_name = train_dataset.classes[label_idx]

    # Convert Tensor to Numpy for plotting: CHW -> HWC
    img_numpy = img_tensor.numpy().transpose((1, 2, 0))

    # Plot on the SPECIFIC axis ax[i]
    axes_flat[i].imshow(img_numpy)
    axes_flat[i].set_title(class_name, fontsize=10)
    axes_flat[i].axis("off")  # Hide axis ticks

plt.tight_layout()
plt.show()

label_counts = Counter([label for _, label in train_dataset.samples])
print("\nImages per class:")
for cls_name, count in label_counts.items():
    print(f"{cls_name}: {count} images")

# Check image shape (should be [3, 100, 100] after resize + ToTensor)
sample_img, sample_label = train_dataset[0]
print(
    f"\nSample image shape: {sample_img.shape}"
)  # Should be torch.Size([3, 100, 100])
print(f"Sample label index: {sample_label}")
print(f"Sample label name: {train_dataset.classes[sample_label]}")

classes = list(label_counts.keys())
counts = list(label_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(classes, counts)
plt.xlabel("Cell Type")
plt.ylabel("Number of Images")
plt.title("Class Distribution in Training Set")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
