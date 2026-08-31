from pathlib import Path

import tensorflow as tf



# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

EPOCHS = 10


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)


# ============================================================
# 3. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PNEUMOVISION MODEL TRAINING")
print("=" * 60)

print("\nClasses:")
print(train_dataset.class_names)

print("\nImage size:")
print(IMAGE_SIZE)

print("\nBatch size:")
print(BATCH_SIZE)

print("\nEpochs:")
print(EPOCHS)


# ============================================================
# 4. IMPROVE DATA PIPELINE PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ],
    name="data_augmentation"
)


# ============================================================
# 6. PRETRAINED MODEL
# ============================================================

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze the pretrained layers initially
base_model.trainable = False


# ============================================================
# 7. BUILD PNEUMOVISION MODEL
# ============================================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = tf.layers.GlobalAveragePooling2D()(x)

x = tf.layers.Dropout(0.4)(x)

x = tf.layers.Dense(128, activation="relu")(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs, name="PneumoVision")


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

print("\nModel summary:")
model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "pneumovision_best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1,
    ),
]

# ============================================================
# 11. CALCULATE CLASS WEIGHTS
# ============================================================

normal_count = len(list((TRAIN_PATH / "NORMAL").glob("*")))
pneumonia_count = len(list((TRAIN_PATH / "PNEUMONIA").glob("*")))

total_count = normal_count + pneumonia_count

class_weight = {
    0: total_count / (2 * normal_count),
    1: total_count / (2 * pneumonia_count),
}

print("\nClass weights:")
print(class_weight)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)


# ============================================================
# 13. SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "pneumovision_final.keras"

model.save(final_model_path)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"\nFinal model saved to:")
print(final_model_path)
