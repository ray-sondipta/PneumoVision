from pathlib import Path

# Dataset location
DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

NORMAL_PATH = DATASET_PATH / "train" / "NORMAL"
PNEUMONIA_PATH = DATASET_PATH / "train" / "PNEUMONIA"

# Count images
normal_count = len(list(NORMAL_PATH.glob("*")))
pneumonia_count = len(list(PNEUMONIA_PATH.glob("*")))

total_count = normal_count + pneumonia_count

# Calculate class weights
normal_weight = total_count / (2 * normal_count)
pneumonia_weight = total_count / (2 * pneumonia_count)

print("=" * 50)
print("       PNEUMOVISION CLASS ANALYSIS")
print("=" * 50)

print(f"\nNORMAL images    : {normal_count}")
print(f"PNEUMONIA images : {pneumonia_count}")
print(f"TOTAL images     : {total_count}")

print("\nClass weights:")
print(f"NORMAL weight    : {normal_weight:.4f}")
print(f"PNEUMONIA weight : {pneumonia_weight:.4f}")

print("\n" + "=" * 50)