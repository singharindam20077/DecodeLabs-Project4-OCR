
# DecodeLabs AI Vision
## AI Project 4: OCR Text Recognition and Image Classification

A Python-based AI application with a Streamlit web interface that extracts text from images and identifies image contents using pretrained AI models.

## Features

- **OCR Text Recognition:** Extract text from uploaded images using Tesseract OCR.
- **Image Preprocessing:** Grayscale conversion, Gaussian blur, and adaptive thresholding using OpenCV.
- **AI Image Recognition:** Classify images using the pretrained MobileNetV2 model.
- **Confidence Scores:** Display OCR confidence and image classification probabilities.
- **Download Results:** Download extracted text as a text file.
- **Web Interface:** Upload images and use both recognition modes in a browser.

## Technologies Used

- Python
- Streamlit
- OpenCV
- Tesseract OCR
- pytesseract
- TensorFlow
- MobileNetV2
- NumPy
- Pillow

## Project Structure

```text
DecodeLabs_Project4_OCR/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── input_images/
│   └── sample_text.png
└── output/
    ├── preprocessed_image.png
    └── recognized_text.txt
```

## Installation

### 1. Install Python

Install Python and ensure it is available through the Windows `py` command.

### 2. Install Tesseract OCR

Install Tesseract OCR separately and verify that its executable is available at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If installed elsewhere, update the path in `app.py` and `main.py`.

### 3. Clone the Repository

```bash
git clone https://github.com/singharindam20077/DecodeLabs-Project4-OCR.git
cd DecodeLabs-Project4-OCR
```

### 4. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### 5. Launch the Web Application

```bash
py -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How to Use

### Text Recognition

1. Select Text Recognition (OCR).
2. Upload a PNG, JPG, JPEG, BMP, or WEBP image.
3. Click **Recognize Text**.
4. View the extracted text and OCR confidence score.
5. Download the extracted text if required.

### Image Recognition

1. Select Image Recognition.
2. Upload an image containing an object or animal.
3. Click **Recognize Image**.
4. View the top five predicted categories and their scores.

The first image-recognition run may download pretrained model weights and requires an internet connection.

## Validation

The sample OCR test achieved an average confidence score of **85.90%**, exceeding the project's 80% benchmark.

Image classification was tested with a cat image, and the model returned cat-related predictions.

These results describe the sample tests performed. Results may vary with image quality and content.

## Output Files

The application saves the following OCR results in the `output/` folder:

- `preprocessed_image.png`
- `recognized_text.txt`

## Notes

- OCR confidence is not the same as overall recognition accuracy.
- Image classification scores are model probabilities, not guaranteed accuracy.
- Tesseract OCR is a separate system dependency and is not installed by `pip install -r requirements.txt`.

## Author

Arindam Singh

## Repository

https://github.com/singharindam20077/DecodeLabs-Project4-OCR
