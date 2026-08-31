from pathlib import Path

import tensorflow as tf


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

TRAIN_PATH = DATASET_PATH / "train"
MODEL_PATH = Path("model/pneumovision_best.keras")
MODEL_DIR = Path("model")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAYERS = 30


# ============================================================
# 2. LOAD DATASETS
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


# ============================================================
# 3. OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 4. LOAD BEST BASELINE MODEL
# ============================================================

print("\nLoading best PneumoVision model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")


# ============================================================
# 5. FIND EFFICIENTNETB0
# ============================================================

base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        base_model = layer
        break


if base_model is None:
    raise ValueError(
        "EfficientNetB0 base model could not be found."
    )


print("\nBase model found:")
print(base_model.name)

print("\nTotal base model layers:")
print(len(base_model.layers))


# ============================================================
# 6. UNFREEZE ONLY THE LAST FEW LAYERS
# ============================================================

base_model.trainable = True

fine_tune_from = max(
    0,
    len(base_model.layers) - FINE_TUNE_LAYERS
)

print("\nFine-tuning from layer:")
print(fine_tune_from)


for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False


# Keep Batch Normalization layers frozen
# because they can become unstable with smaller datasets.

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


# ============================================================
# 7. DISPLAY TRAINABLE LAYERS
# ============================================================

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print("\nTrainable base-model layers:")
print(trainable_count)


# ============================================================
# 8. RECOMPILE WITH A VERY SMALL LEARNING RATE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)


# ============================================================
# 9. CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "pneumovision_finetuned_best.keras",
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        verbose=1,
    ),
]


# ============================================================
# 10. CALCULATE CLASS WEIGHTS
# ============================================================

normal_count = len(
    list((TRAIN_PATH / "NORMAL").glob("*"))
)

pneumonia_count = len(
    list((TRAIN_PATH / "PNEUMONIA").glob("*"))
)

total_count = normal_count + pneumonia_count

class_weight = {
    0: total_count / (2 * normal_count),
    1: total_count / (2 * pneumonia_count),
}

print("\nClass weights:")
print(class_weight)


# ============================================================
# 11. START FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("PNEUMOVISION FINE-TUNING")
print("=" * 60)

print("\nFine-tuning epochs:")
print(FINE_TUNE_EPOCHS)

print("\nStarting fine-tuning...")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)


# ============================================================
# 12. SAVE FINAL FINE-TUNED MODEL
# ============================================================

final_model_path = (
    MODEL_DIR / "pneumovision_finetuned_final.keras"
)

model.save(final_model_path)


print("\n" + "=" * 60)
print("FINE-TUNING COMPLETED")
print("=" * 60)

print("\nFine-tuned model saved to:")
print(final_model_path)