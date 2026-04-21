import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

# Setup Data
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.ImageFolder(root="images/TRAIN", transform=transform)
simple_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Plotting Images
def plot10images(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    _, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 cols

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


# Check image per class for balance number of data per class
label_counts = Counter([label for _, label in train_dataset.samples])
print("\nImages per class:")
for cls_name, count in label_counts.items():
    print(f"{cls_name}: {count} images")

# Check image shape (3, 244, 244)
sample_img, sample_label = train_dataset[0]
print(f"\nSample image shape: {sample_img.shape}")
print(f"Sample label index: {sample_label}")
print(f"Sample label name: {train_dataset.classes[sample_label]}")
