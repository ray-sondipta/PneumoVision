from pathlib import Path
import tensorflow as tf

# Dataset location
DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

# Settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Load training data
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load validation data
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load test data
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Display information
print("\n" + "=" * 50)
print("PNEUMOVISION DATASET")
print("=" * 50)

print("\nClass names:")
print(train_dataset.class_names)

print("\nImage size:")
print(IMAGE_SIZE)

print("\nBatch size:")
print(BATCH_SIZE)

print("\nNumber of training batches:")
print(tf.data.experimental.cardinality(train_dataset).numpy())

print("\nNumber of validation batches:")
print(tf.data.experimental.cardinality(validation_dataset).numpy())

print("\nNumber of test batches:")
print(tf.data.experimental.cardinality(test_dataset).numpy())

print("\n" + "=" * 50)
print("DATA LOADING SUCCESSFUL!")
print("=" * 50)