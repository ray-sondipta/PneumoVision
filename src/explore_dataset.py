from pathlib import Path

# Location of our dataset
DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

# Dataset folders
folders = ["train", "test", "val"]

print("=" * 50)
print("        PNEUMOVISION DATASET ANALYSIS")
print("=" * 50)

for folder in folders:
    folder_path = DATASET_PATH / folder

    normal_path = folder_path / "NORMAL"
    pneumonia_path = folder_path / "PNEUMONIA"

    normal_images = list(normal_path.glob("*"))
    pneumonia_images = list(pneumonia_path.glob("*"))

    print(f"\n{folder.upper()} DATA:")
    print("-" * 30)
    print(f"NORMAL images    : {len(normal_images)}")
    print(f"PNEUMONIA images : {len(pneumonia_images)}")
    print(f"TOTAL images     : {len(normal_images) + len(pneumonia_images)}")

print("\n" + "=" * 50)
print("Analysis completed!")
print("=" * 50)