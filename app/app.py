from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PneumoVision",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    
    .stApp {
        background: #07111f;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        text-align: center;
        padding: 50px 20px 30px 20px;
    }

    .badge {
        color: #38bdf8;
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 20px;
    }

    .title {
        color: white;
        font-size: 64px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .subtitle {
        color: #a9b8ca;
        font-size: 20px;
        max-width: 750px;
        margin: auto;
        line-height: 1.6;
    }

    .card {
        background: #101d30;
        border: 1px solid #20334d;
        border-radius: 18px;
        padding: 25px;
        min-height: 170px;
    }

    .card h3 {
        color: white;
        margin-bottom: 10px;
    }

    .card p {
        color: #9fb0c2;
        line-height: 1.5;
    }

    .metric {
        text-align: center;
        background: #101d30;
        border: 1px solid #20334d;
        border-radius: 18px;
        padding: 22px;
    }

    .metric-value {
        color: #38bdf8;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #9fb0c2;
        font-size: 14px;
    }

    .disclaimer {
        color: #b9b1a4;
        background: #19160f;
        border: 1px solid #51462e;
        border-radius: 14px;
        padding: 20px;
        margin-top: 50px;
        font-size: 13px;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

MODEL_PATH = Path(
    "model/pneumovision_finetuned_best.keras"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# FIND EFFICIENTNET
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "efficientnet" in layer.name.lower():

            base_model = layer
            break


if base_model is None:

    st.error(
        "EfficientNet model could not be found."
    )

    st.stop()


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
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

    st.error(
        "Convolutional layer could not be found."
    )

    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="badge">✦ AI-POWERED CHEST X-RAY ANALYSIS</div>',
    unsafe_allow_html=True
)

st.title("PneumoVision")

st.markdown(
    """
    <div class="subtitle">
    Deep learning-based pneumonia detection from chest
    X-ray images, powered by EfficientNetB0 and
    explainable AI.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN BUTTON
# ============================================================

left, center, right = st.columns(
    [1, 1, 1]
)

with center:

    analyze_clicked = st.button(
        "🫁  Analyze a Chest X-Ray",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "## ✨ Built for Explainable AI"
)

st.caption(
    "More than a prediction — understand what the model sees."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="card">

        <h3>🧠 Deep Learning</h3>

        <p>
        EfficientNetB0 transfer learning model trained
        to classify chest X-ray images into Normal and
        Pneumonia classes.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="card">

        <h3>🎯 Model Performance</h3>

        <p>
        Evaluated using accuracy, precision, recall,
        F1-score and ROC-AUC on unseen test data.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="card">

        <h3>🔥 Explainable AI</h3>

        <p>
        Grad-CAM highlights regions that contributed
        to the model's prediction.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("")
st.markdown("")
st.markdown("## 📊 Model Performance")

st.caption(
    "Fine-tuned PneumoVision model evaluated on the test dataset."
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        """
        <div class="metric">
        <div class="metric-value">87.66%</div>
        <div class="metric-label">Test Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        """
        <div class="metric">
        <div class="metric-value">94.09%</div>
        <div class="metric-label">ROC-AUC</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        """
        <div class="metric">
        <div class="metric-value">92.05%</div>
        <div class="metric-label">Pneumonia Recall</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    st.markdown(
        """
        <div class="metric">
        <div class="metric-value">90.31%</div>
        <div class="metric-label">Pneumonia F1</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("")
st.markdown("")
st.markdown("## 🔬 How PneumoVision Works")

st.caption(
    "From an uploaded X-ray to an explainable prediction."
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    st.info(
        "**01 — Upload**\n\n"
        "Upload a chest X-ray image."
    )


with s2:

    st.info(
        "**02 — Analyze**\n\n"
        "EfficientNetB0 processes the image."
    )


with s3:

    st.info(
        "**03 — Predict**\n\n"
        "The model estimates pneumonia probability."
    )


with s4:

    st.info(
        "**04 — Explain**\n\n"
        "Grad-CAM highlights influential regions."
    )

# ============================================================
# ANALYSIS
# ============================================================

st.markdown("---")

st.header("🫁 Analyze Your X-Ray")

st.write(
    "Upload a chest X-ray image to get an AI-powered prediction."
)

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


if uploaded_file is not None:

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    original_image = tf.keras.utils.load_img(
        uploaded_file,
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


    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner("🔍 Analyzing X-ray..."):

        prediction = model.predict(
            input_image,
            verbose=0
        )[0][0]


    pneumonia_probability = float(prediction) * 100

    normal_probability = (
        1 - float(prediction)
    ) * 100


    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if prediction >= 0.5:

        result = "PNEUMONIA"

        confidence = pneumonia_probability

    else:

        result = "NORMAL"

        confidence = normal_probability


    # ========================================================
    # DISPLAY IMAGE AND RESULT
    # ========================================================

    image_col, result_col = st.columns(
        [1.1, 0.9]
    )


    with image_col:

        st.subheader("Uploaded X-Ray")

        st.image(
            original_image,
            use_container_width=True
        )


    with result_col:

        st.subheader("AI Prediction")


        if result == "PNEUMONIA":

            st.error(
                f"Prediction: {result}"
            )

        else:

            st.success(
                f"Prediction: {result}"
            )


        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        st.metric(
            "Pneumonia Probability",
            f"{pneumonia_probability:.2f}%"
        )


        st.metric(
            "Normal Probability",
            f"{normal_probability:.2f}%"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Prediction Probabilities")

    st.progress(
        int(pneumonia_probability)
    )

    st.caption(
        f"Pneumonia: {pneumonia_probability:.2f}%"
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔥 Explainable AI — Grad-CAM"
    )

    st.write(
        "The heatmap highlights image regions that "
        "contributed to the model's prediction."
    )


    try:

        # ----------------------------------------------------
        # CREATE GRAD-CAM MODEL
        # ----------------------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )


        # ----------------------------------------------------
        # GRADIENT CALCULATION
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, base_output = grad_model(
                input_image,
                training=False
            )

            tape.watch(conv_outputs)

            x = base_output

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

            pneumonia_score = predictions[:, 0]


        grads = tape.gradient(
            pneumonia_score,
            conv_outputs
        )


        # ----------------------------------------------------
        # GENERATE HEATMAP
        # ----------------------------------------------------

        if grads is not None:

            pooled_grads = tf.reduce_mean(
                grads,
                axis=(0, 1, 2)
            )


            conv_outputs = conv_outputs[0]


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

                heatmap = (
                    heatmap / max_value
                )


            heatmap = heatmap.numpy()


            # ------------------------------------------------
            # DISPLAY GRAD-CAM
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(8, 8)
            )


            ax.imshow(
                original_image
            )


            ax.imshow(
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


            ax.set_title(
                f"PneumoVision: {result}"
            )


            ax.axis("off")

            plt.tight_layout()


            st.pyplot(
                fig
            )


            plt.close(fig)


        else:

            st.warning(
                "Grad-CAM could not be generated."
            )


    except Exception as e:

        st.warning(
            "Prediction was successful, but Grad-CAM "
            "could not be generated."
        )

        st.caption(
            f"Grad-CAM details: {e}"
        )
        
st.markdown(
    """
    <div class="disclaimer">

    ⚠️ <b>Medical Disclaimer</b><br><br>

    PneumoVision is an AI-based educational and research
    demonstration. It is not a medical diagnostic tool and
    should not be used as a substitute for evaluation by a
    qualified healthcare professional.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "PneumoVision · EfficientNetB0 · Transfer Learning · Grad-CAM"
)