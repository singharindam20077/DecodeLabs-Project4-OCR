
import io
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="DecodeLabs AI Vision",
    page_icon="🔍",
    layout="wide"
)

# Windows Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111b32);
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #f1f5f9;
}
div.stButton > button[kind="primary"] {
    background: #2563eb;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


def preprocess_image(image):
    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )
    return processed


def recognize_text(image):
    processed = preprocess_image(image)
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        processed, config=config
    )

    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    scores = [
        float(conf)
        for word, conf in zip(data["text"], data["conf"])
        if word.strip() and float(conf) >= 0
    ]

    confidence = (
        sum(scores) / len(scores) if scores else 0
    )

    return text.strip(), confidence, processed


@st.cache_resource(show_spinner="Loading image recognition model...")
def load_image_model():
    # Lazy import keeps OCR working even if TensorFlow fails.
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2
    )
    return MobileNetV2(weights="imagenet")


def recognize_image(image):
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input, decode_predictions
    )

    model = load_image_model()

    resized = image.convert("RGB").resize((224, 224))
    array = np.array(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = preprocess_input(batch)

    predictions = model.predict(batch, verbose=0)
    results = decode_predictions(predictions, top=5)[0]

    return [
        (label.replace("_", " ").title(), float(score))
        for _, label, score in results
    ]


st.title("🔍 DecodeLabs AI Vision")
st.caption(
    "Artificial Intelligence Project 4 | "
    "OCR Text Recognition & Image Classification"
)

st.divider()

mode = st.radio(
    "Choose Recognition Mode",
    ["📝 Text Recognition (OCR)", "🖼️ Image Recognition"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Keep results tied to the selected image and mode.
    file_key = (uploaded_file.name, hash(file_bytes), mode)

    if st.session_state.get("file_key") != file_key:
        st.session_state["file_key"] = file_key
        st.session_state.pop("ocr_result", None)
        st.session_state.pop("image_result", None)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with right:
        if mode == "📝 Text Recognition (OCR)":
            st.subheader("Extract Text")

            if st.button("Recognize Text", type="primary"):
                try:
                    with st.spinner("Extracting text..."):
                        result = recognize_text(image)
                        st.session_state["ocr_result"] = result
                except Exception as error:
                    st.error(f"OCR failed: {error}")

            if "ocr_result" in st.session_state:
                text, confidence, processed = (
                    st.session_state["ocr_result"]
                )

                st.text_area(
                    "Recognized Text",
                    value=text,
                    height=200
                )

                st.metric(
                    "Average OCR Confidence",
                    f"{confidence:.2f}%"
                )

                if confidence >= 80:
                    st.success("80% confidence benchmark PASSED")
                else:
                    st.warning("80% confidence benchmark not met")

                if not text:
                    st.info("No readable text detected.")

                st.download_button(
                    "Download Extracted Text",
                    data=text,
                    file_name="recognized_text.txt",
                    mime="text/plain"
                )

                st.image(
                    processed,
                    caption="Preprocessed Image",
                    use_container_width=True
                )

                # Save outputs for project evidence
                output = Path("output")
                output.mkdir(exist_ok=True)

                cv2.imwrite(
                    str(output / "preprocessed_image.png"),
                    processed
                )

                (output / "recognized_text.txt").write_text(
                    text + "\n",
                    encoding="utf-8"
                )

        else:
            st.subheader("AI Image Recognition")

            st.write(
                "Identify the contents of an image "
                "using a pretrained MobileNetV2 model."
            )

            if st.button("Recognize Image", type="primary"):
                try:
                    with st.spinner("Analyzing image..."):
                        predictions = recognize_image(image)
                        st.session_state["image_result"] = predictions
                except Exception as error:
                    st.error(
                        "Image recognition could not start. "
                        "Check the TensorFlow installation "
                        f"and model download.\n\n{error}"
                    )

            if "image_result" in st.session_state:
                st.subheader("Top 5 Predictions")

                for label, score in st.session_state["image_result"]:
                    st.write(f"**{label}**")
                    st.progress(score)
                    st.caption(f"Model confidence: {score * 100:.2f}%")

                st.info(
                    "Predictions are limited to ImageNet categories. "
                    "Confidence scores are model probabilities, "
                    "not guaranteed accuracy."
                )

st.divider()
st.caption(
    "DecodeLabs | Python · Streamlit · OpenCV · "
    "Tesseract OCR · TensorFlow · MobileNetV2"
)
