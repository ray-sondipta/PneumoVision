from pathlib import Path
from PIL import Image

DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

normal_images = list((DATASET_PATH / "train" / "NORMAL").glob("*"))
pneumonia_images = list((DATASET_PATH / "train" / "PNEUMONIA").glob("*"))

print("PNEUMOVISION IMAGE PROPERTIES")
print("=" * 40)

# Check first NORMAL image
normal_image = Image.open(normal_images[0])

print("\nNORMAL image:")
print("File:", normal_images[0].name)
print("Size:", normal_image.size)
print("Mode:", normal_image.mode)

# Check first PNEUMONIA image
pneumonia_image = Image.open(pneumonia_images[0])

print("\nPNEUMONIA image:")
print("File:", pneumonia_images[0].name)
print("Size:", pneumonia_image.size)
print("Mode:", pneumonia_image.mode)