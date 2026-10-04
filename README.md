# Visual Defect Detection System

## Overview
This lightweight application uses a **pre-trained YOLO11n classification model** (saved as a complete PyTorch object) to classify manufactured casting product images as:

- **Defective**
- **Normal**

The model was trained with transfer learning on a small dataset and the saved .pth file contains the full model, so no architecture reconstruction is required.

## Project Structure
`
defect-detection-project/
+-- app.py
+-- complete_defect_classifier.pth   # provided model file
+-- requirements.txt
+-- README.md
`

## Setup
1. (Optional) Create a virtual environment:
   `ash
   python -m venv venv
   .\\venv\\Scripts\\activate
   `
2. Install dependencies:
   `ash
   pip install -r requirements.txt
   `

## Run Locally
`ash
streamlit run app.py
`
A browser window will open with the UI.

## Usage
1. Click **Browse files** and upload a JPG/PNG image of a casting product.
2. Press **Analyze Image**.
3. The app shows:
   - The uploaded image
   - Predicted class (**Defective** or **Normal**)
   - Confidence percentage
   - Inference time (ms)
   - Device used (CPU or CUDA)

## Classes
- **Defective**
- **Normal**

The UI displays only these user-friendly labels; the original training labels (def_front, ok_front) are hidden.

## Notes
- The app automatically uses **CUDA** if a compatible GPU is available; otherwise it falls back to **CPU**.
- All preprocessing matches the training pipeline (Resize(300,300) ? ToTensor()), with no extra normalization or augmentation.
- Logging (startup, model loading, predictions, errors) is printed to the console for easy debugging.
