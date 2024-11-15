import pytesseract
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set the path to the Tesseract executable (if it's not already in the system PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Update this path if necessary

# Function to preprocess and perform OCR on the image
def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Apply a sharpening filter to improve text visibility
    img = img.filter(ImageFilter.SHARPEN)

    # Convert to NumPy array for further processing
    img = np.array(img)

    # Apply thresholding to convert the image to black and white
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    # Optional: Denoise the image using Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert the processed image back to a PIL image
    img = Image.fromarray(img)

    # Display the processed image (optional)
    plt.imshow(img, cmap='gray')
    plt.title(f'Processed Image: {image_path}')
    plt.show()

    return img

# Function to perform OCR on the processed image
def ocr_from_image(image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(processed_img)

    return text

# Test the function with an image
image_path = 'hand1.jpg'  # Replace with the path to your image file
extracted_text = ocr_from_image(image_path)

# Print the extracted text
print("Extracted Text:")
print(extracted_text)

# Save the extracted text to a text file (optional)
with open('output_text.txt', 'w') as f:
    f.write(extracted_text)
    print("\nExtracted text has been saved to 'output_text.txt'.")
