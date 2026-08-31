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
    page_icon="P",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL BACKGROUND
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 50% -10%,
                #12365a 0%,
                #091827 38%,
                #050d17 75%,
                #030811 100%
            );

        color: #eaf4ff;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       GENERAL TYPOGRAPHY
       ======================================================== */

    html,
    body {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #f4f8fc !important;
        letter-spacing: -0.02em;
    }

    p {
        color: #b9cddd;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #9ab0c2 !important;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        text-align: center;
        padding: 55px 20px 28px 20px;
    }

    .badge {
        display: inline-block;

        color: #7be3f7;

        background: rgba(43, 173, 213, 0.12);

        border: 1px solid rgba(74, 195, 230, 0.32);

        border-radius: 999px;

        padding: 7px 16px;

        font-size: 12px;

        font-weight: 700;

        letter-spacing: 1.3px;

        margin-bottom: 22px;

        box-shadow:
            0 3px 14px rgba(30, 174, 213, 0.08);
    }

    .hero-title {
        color: #f7fbff;

        font-size: 64px;

        font-weight: 800;

        line-height: 1.05;

        letter-spacing: -2px;

        margin-bottom: 18px;
    }

    .hero-title span {
        color: #4fd1e8;
    }

    .subtitle {
        color: #c7d9e8;

        font-size: 19px;

        max-width: 760px;

        margin: auto;

        line-height: 1.65;
    }

    .subtitle strong {
        color: #f1fbff;
    }


    /* ========================================================
       MAIN ANALYZE BUTTON
       ======================================================== */

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #20b9df 0%,
                #168db8 55%,
                #116f98 100%
            ) !important;

        color: #ffffff !important;

        border: 1px solid #5bdcf5 !important;

        border-radius: 10px !important;

        font-size: 15px !important;

        font-weight: 700 !important;

        height: 50px !important;

        letter-spacing: 0.15px;

        box-shadow:
            0 6px 20px rgba(24, 181, 221, 0.28),
            0 0 0 1px rgba(91, 220, 245, 0.08) !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            filter 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        filter: brightness(1.08);

        box-shadow:
            0 10px 28px rgba(24, 181, 221, 0.38),
            0 0 18px rgba(55, 211, 241, 0.16) !important;
    }

    .stButton > button:active {
        transform: translateY(0px);
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .card {
        background:
            linear-gradient(
                145deg,
                rgba(17, 39, 61, 0.96),
                rgba(9, 24, 40, 0.96)
            );

        border: 1px solid #1e3c57;

        border-radius: 16px;

        padding: 26px;

        min-height: 175px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.16);

        transition:
            border-color 0.2s ease,
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .card:hover {
        border-color: #2d6685;

        transform: translateY(-2px);

        box-shadow:
            0 15px 34px rgba(0, 0, 0, 0.22);
    }

    .card h3 {
        color: #edf7ff !important;

        margin-bottom: 12px;

        font-size: 20px;

        font-weight: 700;
    }

    .card p {
        color: #b4c7d8;

        line-height: 1.65;

        font-size: 14px;

        margin-bottom: 0;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric {
        text-align: center;

        background:
            linear-gradient(
                145deg,
                rgba(16, 39, 61, 0.98),
                rgba(8, 23, 38, 0.98)
            );

        border: 1px solid #21425e;

        border-radius: 15px;

        padding: 23px 15px;

        box-shadow:
            0 10px 28px rgba(0, 0, 0, 0.14);
    }

    .metric-value {
        color: #51d4f2;

        font-size: 29px;

        font-weight: 800;

        letter-spacing: -0.5px;
    }

    .metric-label {
        color: #a1b6c7;

        font-size: 13px;

        margin-top: 5px;
    }


    /* ========================================================
       HOW IT WORKS
       ======================================================== */

    [data-testid="stAlert"] {
        background: #0d2237;

        border: 1px solid #204663;

        color: #c5d8e8;

        border-radius: 13px;

        box-shadow:
            0 7px 20px rgba(0, 0, 0, 0.10);
    }

    [data-testid="stAlert"] p {
        color: #b9cddd !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(10, 28, 45, 0.70);

        border: 1px solid #23445e;

        border-radius: 14px;

        padding: 8px;

        box-shadow:
            0 8px 22px rgba(0, 0, 0, 0.10);
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0b1d2e !important;

        border: 1px dashed #3a6983 !important;

        border-radius: 11px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #b5c9d9 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #a8bfd0 !important;
    }


    /* ========================================================
       IMAGE
       ======================================================== */

    [data-testid="stImage"] img {
        border-radius: 14px;

        border: 1px solid #23445e;

        box-shadow:
            0 10px 25px rgba(0, 0, 0, 0.16);
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    [data-testid="stProgress"] > div {
        background-color: #132b40 !important;

        border-radius: 20px;
    }

    [data-testid="stProgress"] > div > div {
        background:
            linear-gradient(
                90deg,
                #1a9bc0,
                #53d8f2
            ) !important;

        border-radius: 20px;
    }


    /* ========================================================
       STREAMLIT METRIC COMPONENT
       ======================================================== */

    [data-testid="stMetric"] {
        background: #0c2032;

        border: 1px solid #21425e;

        border-radius: 13px;

        padding: 15px;

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.10);
    }

    [data-testid="stMetricLabel"] {
        color: #9eb3c4 !important;
    }

    [data-testid="stMetricValue"] {
        color: #eaf7ff !important;
    }


    /* ========================================================
       RESULT BOXES
       ======================================================== */

    [data-testid="stAlert"][kind="error"] {
        background: rgba(111, 39, 48, 0.25);

        border: 1px solid rgba(214, 91, 103, 0.45);
    }

    [data-testid="stAlert"][kind="success"] {
        background: rgba(27, 100, 88, 0.25);

        border: 1px solid rgba(67, 180, 153, 0.45);
    }


    /* ========================================================
       HEADERS
       ======================================================== */

    .stHeader {
        color: #f4f8fc !important;
    }


    /* ========================================================
       DISCLAIMER
       ======================================================== */

    .disclaimer {
        color: #a9bac7;

        background:
            linear-gradient(
                145deg,
                rgba(22, 32, 43, 0.92),
                rgba(13, 23, 34, 0.92)
            );

        border: 1px solid #344554;

        border-radius: 13px;

        padding: 19px 21px;

        margin-top: 45px;

        font-size: 12px;

        line-height: 1.65;
    }

    .disclaimer strong {
        color: #dce8ef;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #7690a4;

        font-size: 12px;

        padding-top: 5px;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #071421;
    }

    ::-webkit-scrollbar-thumb {
        background: #28465d;

        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #38647e;
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
    '<div class="badge">AI-POWERED CHEST X-RAY ANALYSIS</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-title">
        Pneumo<span>Vision</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Deep learning-based pneumonia detection from chest
        X-ray images, powered by
        <strong>EfficientNetB0</strong> and
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
        "Analyze a Chest X-Ray",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    "## Built for Explainable AI"
)

st.caption(
    "More than a prediction — understand what the model sees."
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        """
        <div class="card">

        <h3>Deep Learning</h3>

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

        <h3>Model Performance</h3>

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

        <h3>Explainable AI</h3>

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

st.markdown(
    "## Model Performance"
)

st.caption(
    "Fine-tuned PneumoVision model evaluated on the test dataset."
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.markdown(
        """
        <div class="metric">

        <div class="metric-value">
        87.66%
        </div>

        <div class="metric-label">
        Test Accuracy
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m2:

    st.markdown(
        """
        <div class="metric">

        <div class="metric-value">
        94.09%
        </div>

        <div class="metric-label">
        ROC-AUC
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m3:

    st.markdown(
        """
        <div class="metric">

        <div class="metric-value">
        92.05%
        </div>

        <div class="metric-label">
        Pneumonia Recall
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with m4:

    st.markdown(
        """
        <div class="metric">

        <div class="metric-value">
        90.31%
        </div>

        <div class="metric-label">
        Pneumonia F1
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown("")
st.markdown("")

st.markdown(
    "## How PneumoVision Works"
)

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

st.header(
    "Analyze Your X-Ray"
)

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

    with st.spinner(
        "Analyzing X-ray..."
    ):

        prediction = model.predict(
            input_image,
            verbose=0
        )[0][0]


    pneumonia_probability = (
        float(prediction) * 100
    )

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

        st.subheader(
            "Uploaded X-Ray"
        )

        st.image(
            original_image,
            use_container_width=True
        )


    with result_col:

        st.subheader(
            "AI Prediction"
        )


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

    st.subheader(
        "Prediction Probabilities"
    )

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
        "Explainable AI — Grad-CAM"
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


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

    <strong>Medical Disclaimer</strong><br><br>

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

st.markdown(
    """
    <div class="footer">
        PneumoVision · EfficientNetB0 · Transfer Learning · Grad-CAM
    </div>
    """,
    unsafe_allow_html=True
)