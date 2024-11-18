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
    img = img.convert('L')
    img = img.filter(ImageFilter.SHARPEN)
    img = np.array(img)
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return Image.fromarray(img)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    image_file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(file_path)

    processed_img = preprocess_image(file_path)
    text = pytesseract.image_to_string(processed_img)

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)
