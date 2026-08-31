from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

MODEL_PATH = Path(
    "model/pneumovision_finetuned_best.keras"
)

IMAGE_SIZE = (224, 224)

OUTPUT_DIR = Path("images/gradcam")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading PneumoVision model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 3. FIND EFFICIENTNET BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            base_model = layer
            break


if base_model is None:

    raise ValueError(
        "EfficientNet base model could not be found."
    )


print("\nBase model found:")
print(base_model.name)


# ============================================================
# 4. FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(base_model.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        break


if last_conv_layer is None:

    raise ValueError(
        "Convolutional layer could not be found."
    )


print("\nUsing convolutional layer:")
print(last_conv_layer.name)


# ============================================================
# 5. LOAD IMAGE
# ============================================================

image_path = input(
    "\nEnter the path of an X-ray image: "
).strip().strip('"')


if not Path(image_path).exists():

    raise FileNotFoundError(
        "\nERROR: Image not found."
    )


original_image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE,
    color_mode="rgb"
)


image_array = tf.keras.utils.img_to_array(
    original_image
)


input_image = np.expand_dims(
    image_array,
    axis=0
)


input_image = tf.cast(
    input_image,
    tf.float32
)


# ============================================================
# 6. CREATE GRAD-CAM MODEL
# ============================================================

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)


# ============================================================
# 7. CALCULATE GRADIENTS
# ============================================================

with tf.GradientTape() as tape:

    conv_outputs, base_output = grad_model(
        input_image,
        training=False
    )

    tape.watch(conv_outputs)

    # Reproduce the classification head
    x = base_output

    # Find layers after EfficientNet
    base_index = model.layers.index(
        base_model
    )

    head_layers = model.layers[
        base_index + 1:
    ]

    for layer in head_layers:

        x = layer(
            x,
            training=False
        )

    predictions = x

    pneumonia_probability = predictions[:, 0]


# ============================================================
# 8. CALCULATE GRADIENTS
# ============================================================

grads = tape.gradient(
    pneumonia_probability,
    conv_outputs
)


if grads is None:

    raise ValueError(
        "Gradients are None. "
        "Grad-CAM could not calculate gradients."
    )


# ============================================================
# 9. CALCULATE CHANNEL IMPORTANCE
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


conv_outputs = conv_outputs[0]


# ============================================================
# 10. GENERATE HEATMAP
# ============================================================

heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)


heatmap = tf.maximum(
    heatmap,
    0
)


max_value = tf.reduce_max(
    heatmap
)


if float(max_value) > 0:

    heatmap = heatmap / max_value


heatmap = heatmap.numpy()


# ============================================================
# 11. PREDICTION
# ============================================================

probability = float(
    pneumonia_probability.numpy()[0]
)


if probability >= 0.5:

    prediction = "PNEUMONIA"

    confidence = probability * 100

else:

    prediction = "NORMAL"

    confidence = (
        1 - probability
    ) * 100


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "PNEUMOVISION - GRAD-CAM EXPLANATION"
)

print("=" * 60)

print(
    f"\nPrediction: {prediction}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    f"Pneumonia Probability: "
    f"{probability * 100:.2f}%"
)


# ============================================================
# 13. SAVE GRAD-CAM IMAGE
# ============================================================

output_path = (
    OUTPUT_DIR /
    "pneumovision_gradcam.jpg"
)


plt.figure(
    figsize=(8, 8)
)


# Original X-ray
plt.imshow(
    original_image
)


# Heatmap overlay
plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.4,
    extent=(
        0,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        0
    )
)


plt.title(
    f"PneumoVision: {prediction} "
    f"({confidence:.2f}%)"
)


plt.axis("off")

plt.tight_layout()


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 14. COMPLETION MESSAGE
# ============================================================

print(
    "\nGrad-CAM image saved to:"
)

print(output_path)

print(
    "\nGrad-CAM completed successfully!"
)