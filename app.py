from flask import Flask, request, jsonify, render_template
import pytesseract
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    img = img.filter(ImageFilter.SHARPEN)  # Enhance sharpness
    img = np.array(img)

    # Adaptive thresholding for better binarization
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return Image.fromarray(img)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        image_file = request.files['image']
        print("Received file:", image_file.filename)  # Debugging statement
        file_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(file_path)
        print(f"File saved at: {file_path}")  # Debugging statement

        processed_img = preprocess_image(file_path)

        # Visualize the processed image (for debugging)
        processed_img.save('processed_image.png')  # Save the processed image to check

        # Extract text with layout retention (using --psm 6)
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        print("OCR Output:", text)  # Debugging statement

        # Return text wrapped in HTML for styling
        return jsonify({'text': f'<div class="ocr-text">{text}</div>'})

    except Exception as e:
        print(f"Error during image processing: {e}")  # Debugging statement
        return jsonify({'error': 'Error processing image!'}), 500

if __name__ == '__main__':
    app.run(debug=True)
