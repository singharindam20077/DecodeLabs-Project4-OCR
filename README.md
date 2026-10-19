
# DecodeLabs AI Project 4: OCR Text Recognition

## Project Overview
This project implements Optical Character Recognition (OCR) using Python, Tesseract OCR, and OpenCV. It extracts text from an input image and saves the recognized text as a file.

## Objectives
- Extract readable text from images.
- Apply image preprocessing to improve recognition.
- Use a pretrained OCR engine.
- Display the extracted text and confidence score.
- Validate the result against an 80% confidence benchmark.

## Technologies Used
- Python
- OpenCV
- pytesseract
- Tesseract OCR

## Methodology
1. Load the input image.
2. Convert the image to grayscale.
3. Apply Gaussian blur to reduce noise.
4. Use adaptive thresholding to improve contrast.
5. Extract text using Tesseract OCR.
6. Calculate the average word confidence.
7. Save the processed image and recognized text.

## Project Structure
```text
DecodeLabs_Project4_OCR/
├── input_images/
│   └── sample_text.png
├── output/
│   ├── preprocessed_image.png
│   └── recognized_text.txt
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install Python packages
```bash
py -m pip install -r requirements.txt
```

### 2. Install Tesseract OCR
Install the Tesseract OCR engine separately and ensure the path in `main.py` matches your installation.

### 3. Run the application
```bash
py main.py
```

## Results
The application successfully extracted text from the sample image.

Observed average OCR confidence during testing: **85.90%**.

The result exceeded the project's 80% confidence benchmark for this test image.

## Output Files
- `output/preprocessed_image.png`: Preprocessed image.
- `output/recognized_text.txt`: Extracted text.

## Limitations
OCR accuracy may decrease with blurry images, handwriting, unusual fonts, poor lighting, or tilted text.

Confidence scores estimate the OCR engine's certainty and do not guarantee perfect recognition.

## Conclusion
This project demonstrates a basic OCR pipeline using image preprocessing and a pretrained recognition engine to extract machine-readable text from images.

## Author
Student project for the DecodeLabs Artificial Intelligence Industrial Training Kit.
