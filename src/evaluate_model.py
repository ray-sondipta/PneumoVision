from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = Path(r"C:\PROJECTS\chest_xray")

TEST_PATH = DATASET_PATH / "test"

MODEL_PATH = Path("model/pneumovision_best.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\nLoading PneumoVision model...")

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(test_dataset)

probabilities = predictions.ravel()

predicted_classes = (probabilities >= 0.5).astype(int)


# ============================================================
# 5. GET TRUE LABELS
# ============================================================

true_classes = np.concatenate(
    [labels.numpy().ravel() for images, labels in test_dataset]
).astype(int)


# ============================================================
# 6. CLASS NAMES
# ============================================================

class_names = test_dataset.class_names

print("\nClass names:")
print(class_names)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        digits=4,
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes,
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 9. ROC-AUC
# ============================================================

auc_score = roc_auc_score(
    true_classes,
    probabilities,
)

print(f"\nROC-AUC: {auc_score:.4f}")


# ============================================================
# 10. DISPLAY CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot()

plt.title("PneumoVision - Confusion Matrix")
plt.tight_layout()
plt.show()