from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Location of our dataset
DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

# Select some images
normal_images = list((DATASET_PATH / "train" / "NORMAL").glob("*"))[:2]
pneumonia_images = list((DATASET_PATH / "train" / "PNEUMONIA").glob("*"))[:2]

# Create a 2 x 2 display
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Display NORMAL images
for i, image_path in enumerate(normal_images):
    image = Image.open(image_path)

    axes[0, i].imshow(image, cmap="gray")
    axes[0, i].set_title("NORMAL")
    axes[0, i].axis("off")

# Display PNEUMONIA images
for i, image_path in enumerate(pneumonia_images):
    image = Image.open(image_path)

    axes[1, i].imshow(image, cmap="gray")
    axes[1, i].set_title("PNEUMONIA")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()