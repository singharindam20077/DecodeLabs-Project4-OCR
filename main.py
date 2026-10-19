
from pathlib import Path

import cv2
import pytesseract

# Tesseract OCR installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def preprocess_image(image_path):
    """Prepare an image for OCR recognition."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot open image: {image_path}")

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Reduce image noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 3: Adaptive thresholding
    processed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


def main():
    image_path = Path("input_images/sample_text.png")

    # Preprocess the input image
    processed = preprocess_image(image_path)

    # Extract text using Tesseract OCR
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        processed,
        config=config
    )

    # Calculate average word confidence
    data = pytesseract.image_to_data(
        processed,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    confidences = []

    for value in data["conf"]:
        try:
            score = float(value)
            if score >= 0:
                confidences.append(score)
        except (ValueError, TypeError):
            continue

    confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )

    # Create output folder
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save preprocessed image
    cv2.imwrite(
        str(output_dir / "preprocessed_image.png"),
        processed
    )

    # Save extracted text
    (output_dir / "recognized_text.txt").write_text(
        text.strip() + "\n",
        encoding="utf-8"
    )

    # Display results
    print("\n" + "=" * 45)
    print("DECODELABS PROJECT 4")
    print("OCR TEXT RECOGNITION")
    print("=" * 45)

    print("\n--- RECOGNIZED TEXT ---")
    print(text.strip() or "No text detected")

    print("\n--- VALIDATION ---")
    print(f"Average OCR confidence: {confidence:.2f}%")

    if confidence >= 80:
        print("80% confidence benchmark: PASSED")
    else:
        print("80% confidence benchmark: NOT MET")

    print("\n--- OUTPUT FILES ---")
    print("output/preprocessed_image.png")
    print("output/recognized_text.txt")
    print("=" * 45)


if __name__ == "__main__":
    main()
