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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 10%,
                rgba(35, 120, 150, 0.16),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #06131f 0%,
                #081a29 50%,
                #06131f 100%
            );
        color: #f4f8fb;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0a1d2b 0%,
                #071724 100%
            );
        border-right: 1px solid #18394b;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f5f9fc;
    }

    section[data-testid="stSidebar"] p {
        color: #a8bdca;
    }

    .sidebar-divider {
        height: 1px;
        background: #204456;
        margin: 28px 0;
    }

    .sidebar-label {
        color: #8fa9b8;
        font-size: 13px;
        margin-top: 18px;
        margin-bottom: 4px;
    }

    .sidebar-value {
        color: #f5fbff;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .sidebar-section-title {
        color: #f5f9fc;
        font-size: 16px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .sidebar-detail {
        color: #b6c8d3;
        font-size: 13px;
        line-height: 1.8;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero-wrapper {
        background:
            linear-gradient(
                135deg,
                rgba(18, 53, 70, 0.95),
                rgba(8, 28, 42, 0.97)
            );
        border: 1px solid #24586c;
        border-radius: 24px;
        padding: 52px 58px;
        margin-bottom: 28px;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);
    }

    .hero-icon {
        font-size: 46px;
        margin-bottom: 20px;
    }

    .eyebrow {
        color: #52d6ee;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .hero-title {
        color: #f7fbfd;
        font-size: 58px;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -1.5px;
        margin-bottom: 20px;
    }

    .hero-title span {
        color: #49cfe8;
    }

    .hero-description {
        color: #b7cbd6;
        font-size: 18px;
        line-height: 1.75;
        max-width: 820px;
        margin-bottom: 26px;
    }

    .tech-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
    }

    .tech-tag {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        border: 1px solid #2d7187;
        background: rgba(31, 105, 126, 0.18);
        color: #70def2;
        font-size: 12px;
        font-weight: 700;
    }

    /* ======================================================
       DISCLAIMER
       ====================================================== */

    .disclaimer {
        background: rgba(38, 36, 24, 0.72);
        border: 1px solid #75612b;
        border-left: 4px solid #e8bd4b;
        border-radius: 14px;
        padding: 18px 22px;
        margin: 24px 0 34px 0;
        color: #d7d0b9;
        font-size: 13px;
        line-height: 1.7;
    }

    .disclaimer-title {
        color: #f1d47b;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 4px;
    }

    /* ======================================================
       SECTION HEADINGS
       ====================================================== */

    .section-title {
        color: #f5f9fc;
        font-size: 30px;
        font-weight: 750;
        letter-spacing: -0.5px;
        margin-top: 42px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #829eae;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* ======================================================
       FEATURE CARDS
       ====================================================== */

    .feature-card {
        background:
            linear-gradient(
                145deg,
                rgba(17, 43, 58, 0.96),
                rgba(10, 28, 40, 0.96)
            );
        border: 1px solid #20495d;
        border-radius: 18px;
        padding: 25px;
        min-height: 180px;
        transition: 0.2s ease;
    }

    .feature-card:hover {
        border-color: #3a8298;
    }

    .feature-title {
        color: #f4f9fc;
        font-size: 19px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .feature-text {
        color: #9db5c3;
        font-size: 14px;
        line-height: 1.7;
    }

    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                #102d3e,
                #0b2231
            );
        border: 1px solid #24566d;
        border-radius: 17px;
        padding: 24px 15px;
        text-align: center;
        min-height: 115px;
    }

    .metric-value {
        color: #5edcf1;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .metric-label {
        color: #a7bdc9;
        font-size: 12px;
        font-weight: 500;
    }

    /* ======================================================
       WORKFLOW CARDS
       ====================================================== */

    .workflow-card {
        background: rgba(12, 35, 49, 0.82);
        border: 1px solid #1e4658;
        border-radius: 16px;
        padding: 21px;
        min-height: 135px;
    }

    .workflow-number {
        color: #52d6ee;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .workflow-title {
        color: #f5f9fc;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .workflow-text {
        color: #98b1bf;
        font-size: 13px;
        line-height: 1.55;
    }

    /* ======================================================
       ANALYSIS AREA
       ====================================================== */

    .analysis-box {
        background:
            linear-gradient(
                145deg,
                rgba(14, 42, 57, 0.96),
                rgba(8, 27, 39, 0.96)
            );
        border: 1px solid #24566c;
        border-radius: 20px;
        padding: 28px;
        margin-top: 25px;
    }

    .result-box {
        border-radius: 16px;
        padding: 22px;
        margin-top: 12px;
    }

    .result-positive {
        background: rgba(127, 42, 42, 0.16);
        border: 1px solid #8b4c4c;
    }

    .result-negative {
        background: rgba(42, 110, 87, 0.16);
        border: 1px solid #3e7d69;
    }

    .result-title {
        font-size: 14px;
        color: #9eb5c1;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 27px;
        font-weight: 800;
        color: #f5f9fc;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        border-top: 1px solid #1d3d4e;
        margin-top: 60px;
        padding-top: 25px;
        text-align: center;
    }

    .footer-name {
        color: #eaf5f8;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .footer-course {
        color: #58cfe5;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .footer-text {
        color: #7893a1;
        font-size: 12px;
        line-height: 1.6;
    }

    /* ======================================================
       STREAMLIT BUTTON
       ====================================================== */

    div.stButton > button {
        background:
            linear-gradient(
                135deg,
                #39c7df,
                #179bb8
            );
        color: #ffffff;
        border: none;
        border-radius: 12px;
        height: 52px;
        font-size: 15px;
        font-weight: 800;
        box-shadow:
            0 8px 25px rgba(30, 180, 210, 0.20);
    }

    div.stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #4bd5eb,
                #1ba9c5
            );
        color: white;
        border: none;
    }

    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(13, 39, 53, 0.75);
        border: 1px dashed #347b91;
        border-radius: 15px;
        padding: 10px;
    }

    /* ======================================================
       GENERAL TEXT
       ====================================================== */

    h1, h2, h3, h4 {
        color: #f5f9fc !important;
    }

    .stCaption {
        color: #829eae !important;
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
# FIND EFFICIENTNET BASE MODEL
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
# SIDEBAR — MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2 style="margin-bottom:5px;">
        Model Information
        </h2>

        <p style="color:#63d4e9;">
        EfficientNetB0 transfer-learning classifier
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section-title">Model Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-detail">
        <b>Architecture:</b> EfficientNetB0<br>
        <b>Approach:</b> Transfer Learning<br>
        <b>Input:</b> 224 × 224 RGB<br>
        <b>Classes:</b> Normal / Pneumonia
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section-title">Test Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-label">Test Accuracy</div>
        <div class="sidebar-value">87.66%</div>

        <div class="sidebar-label">ROC-AUC</div>
        <div class="sidebar-value">94.09%</div>

        <div class="sidebar-label">Pneumonia Recall</div>
        <div class="sidebar-value">92.05%</div>

        <div class="sidebar-label">Pneumonia Precision</div>
        <div class="sidebar-value">88.64%</div>

        <div class="sidebar-label">Pneumonia F1-Score</div>
        <div class="sidebar-value">90.31%</div>

        <div class="sidebar-label">Macro F1-Score</div>
        <div class="sidebar-value">86.66%</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section-title">Input Requirements</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-detail">
        • Chest X-ray image<br>
        • JPG, JPEG or PNG<br>
        • Recommended: clear frontal X-ray
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero-wrapper">

        <div class="hero-icon">🫁</div>

        <div class="eyebrow">
            Deep Learning · Chest X-Ray Analysis
        </div>

        <div class="hero-title">
            Pneumo<span>Vision</span>
        </div>

        <div class="hero-description">
            An AI-powered chest X-ray analysis system that
            uses an EfficientNetB0 transfer-learning model
            to classify images as Normal or Pneumonia,
            with Grad-CAM providing visual insight into
            the model's prediction.
        </div>

        <div class="tech-tags">
            <span class="tech-tag">TensorFlow</span>
            <span class="tech-tag">EfficientNetB0</span>
            <span class="tech-tag">Transfer Learning</span>
            <span class="tech-tag">Grad-CAM</span>
            <span class="tech-tag">Streamlit</span>
        </div>

    </div>
    """
)


# ============================================================
# DISCLAIMER
# ============================================================

st.html(
    """
    <div class="disclaimer">

        <div class="disclaimer-title">
            Educational prototype only.
        </div>

        This application is not clinically validated and
        must not replace examination, diagnosis or advice
        from a qualified medical professional.

    </div>
    """
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

button_left, button_center, button_right = st.columns(
    [1.2, 1, 1.2]
)

with button_center:

    analyze_clicked = st.button(
        "Analyze a Chest X-Ray",
        use_container_width=True,
        type="primary"
    )


if analyze_clicked:

    st.html(
        """
        <script>
        window.parent.document.getElementById(
            "analysis"
        )?.scrollIntoView({
            behavior: "smooth"
        });
        </script>
        """
    )


# ============================================================
# FEATURES
# ============================================================

st.html(
    """
    <div class="section-title">
        Built for Explainable AI
    </div>

    <div class="section-subtitle">
        More than a prediction — understand what the model sees.
    </div>
    """
)


feature1, feature2, feature3 = st.columns(3)


with feature1:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-title">
                Deep Learning
            </div>

            <div class="feature-text">
                EfficientNetB0 transfer learning model trained
                to classify chest X-ray images into Normal and
                Pneumonia classes.
            </div>

        </div>
        """
    )


with feature2:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-title">
                Model Performance
            </div>

            <div class="feature-text">
                Evaluated using accuracy, precision, recall,
                F1-score and ROC-AUC on unseen test data.
            </div>

        </div>
        """
    )


with feature3:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-title">
                Explainable AI
            </div>

            <div class="feature-text">
                Grad-CAM highlights image regions that
                contributed to the model's prediction,
                making the result more interpretable.
            </div>

        </div>
        """
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.html(
    """
    <div class="section-title">
        Model Performance
    </div>

    <div class="section-subtitle">
        Fine-tuned PneumoVision model evaluated on unseen test data.
    </div>
    """
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-value">87.66%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """
    )


with m2:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-value">94.09%</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """
    )


with m3:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-value">92.05%</div>
            <div class="metric-label">Pneumonia Recall</div>
        </div>
        """
    )


with m4:

    st.html(
        """
        <div class="metric-card">
            <div class="metric-value">90.31%</div>
            <div class="metric-label">Pneumonia F1-Score</div>
        </div>
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.html(
    """
    <div class="section-title">
        How PneumoVision Works
    </div>

    <div class="section-subtitle">
        From an uploaded X-ray to an explainable prediction.
    </div>
    """
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    st.html(
        """
        <div class="workflow-card">

            <div class="workflow-number">01</div>

            <div class="workflow-title">
                Upload
            </div>

            <div class="workflow-text">
                Upload a chest X-ray image in JPG,
                JPEG or PNG format.
            </div>

        </div>
        """
    )


with s2:

    st.html(
        """
        <div class="workflow-card">

            <div class="workflow-number">02</div>

            <div class="workflow-title">
                Analyze
            </div>

            <div class="workflow-text">
                EfficientNetB0 processes the image
                using the trained feature extractor.
            </div>

        </div>
        """
    )


with s3:

    st.html(
        """
        <div class="workflow-card">

            <div class="workflow-number">03</div>

            <div class="workflow-title">
                Predict
            </div>

            <div class="workflow-text">
                The model estimates the probability
                of pneumonia from the X-ray.
            </div>

        </div>
        """
    )


with s4:

    st.html(
        """
        <div class="workflow-card">

            <div class="workflow-number">04</div>

            <div class="workflow-title">
                Explain
            </div>

            <div class="workflow-text">
                Grad-CAM highlights influential
                regions used by the model.
            </div>

        </div>
        """
    )


# ============================================================
# ANALYSIS SECTION
# ============================================================

st.markdown(
    '<div id="analysis"></div>',
    unsafe_allow_html=True
)

st.html(
    """
    <div class="section-title">
        Analyze Your X-Ray
    </div>

    <div class="section-subtitle">
        Upload a chest X-ray to receive an AI-powered prediction.
    </div>
    """
)


uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

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


    with st.spinner("Analyzing X-ray..."):

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
    # RESULT
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

            st.html(
                f"""
                <div class="result-box result-positive">

                    <div class="result-title">
                        Prediction
                    </div>

                    <div class="result-value">
                        {result}
                    </div>

                </div>
                """
            )

        else:

            st.html(
                f"""
                <div class="result-box result-negative">

                    <div class="result-title">
                        Prediction
                    </div>

                    <div class="result-value">
                        {result}
                    </div>

                </div>
                """
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
    # PROBABILITY
    # ========================================================

    st.subheader("Prediction Probabilities")

    st.progress(
        int(pneumonia_probability)
    )

    st.caption(
        f"Pneumonia probability: "
        f"{pneumonia_probability:.2f}%"
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
            # DISPLAY
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
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <div class="footer-name">
            Developed by Sondipta Ray
        </div>

        <div class="footer-course">
            B.Tech CSE — Artificial Intelligence & Machine Learning
        </div>

        <div class="footer-text">
            Designed, developed and implemented independently,
            including the machine learning pipeline,
            model fine-tuning, evaluation, Grad-CAM explainability
            and Streamlit deployment.
        </div>

        <div class="footer-text" style="margin-top:12px;">
            PneumoVision · EfficientNetB0 · Transfer Learning · Grad-CAM
        </div>

    </div>
    """
)