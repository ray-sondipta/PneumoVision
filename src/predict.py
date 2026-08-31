from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

MODEL_PATH = Path(
    "model/pneumovision_finetuned_best.keras"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading PneumoVision model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 3. IMAGE PREDICTION FUNCTION
# ============================================================

def predict_pneumonia(image_path):

    # Load X-ray image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    # Convert image to NumPy array
    image_array = tf.keras.utils.img_to_array(
        image
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Make prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    # Convert probability to percentage
    pneumonia_probability = prediction * 100

    normal_probability = (
        1 - prediction
    ) * 100

    # Determine prediction
    if prediction >= 0.5:

        result = "PNEUMONIA"

        confidence = pneumonia_probability

    else:

        result = "NORMAL"

        confidence = normal_probability

    return (
        result,
        confidence,
        pneumonia_probability
    )


# ============================================================
# 4. ASK USER FOR IMAGE
# ============================================================

image_path = input(
    "\nEnter the path of an X-ray image: "
).strip().strip('"')


# ============================================================
# 5. CHECK WHETHER IMAGE EXISTS
# ============================================================

if not Path(image_path).exists():

    print(
        "\nERROR: Image not found."
    )

else:

    result, confidence, pneumonia_probability = (
        predict_pneumonia(image_path)
    )

    # ========================================================
    # 6. DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "PNEUMOVISION PREDICTION"
    )

    print("=" * 60)

    print(
        f"\nPrediction: {result}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"Pneumonia Probability: "
        f"{pneumonia_probability:.2f}%"
    )

    print("=" * 60)