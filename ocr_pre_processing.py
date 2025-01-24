import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
from termcolor import colored


def show(image1, image2, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
    ax1.title.set_text("Original Image")
    ax1.imshow(image1)
    ax2.title.set_text(title)
    ax2.imshow(image2)
    plt.show()


def preprocess_image_for_ocr(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Step 1: Resize the image to a reasonable size while maintaining aspect ratio
    height, width = image.shape[:2]
    max_dimension = 1500
    scale = min(max_dimension / width, max_dimension / height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply noise reduction
    # Gaussian blur helps remove high-frequency noise
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Increase contrast using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # Step 5: Thresholding to create binary image
    # Use adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        contrasted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2,  # Constant subtracted from mean
    )

    # Step 6: Morphological operations to clean up the image
    # Create a kernel for morphological operations
    kernel = np.ones((1, 1), np.uint8)

    # Remove small noise with erosion followed by dilation (opening)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Connect nearby text components with dilation followed by erosion (closing)
    final = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return final


def deskew_image(image):
    # Find all non-zero points in the image
    coords = np.column_stack(np.where(image > 0))

    # Calculate the minimum rotated rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # The angle range is [-90,0), we need to handle the cases
    if angle < -45:
        angle = 90 + angle

    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def remove_background(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = hsv[..., 2]

    thresh_val, _ = cv2.threshold(
        brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.threshold(brightness, thresh_val, 255, cv2.THRESH_BINARY)[1]

    result = cv2.bitwise_and(image, image, mask=mask)

    return result


if __name__ == "__main__":
    image_path = "images/1.jpg"

    preprocessed = preprocess_image_for_ocr(image_path)
    deskewed = deskew_image(preprocessed)

    original = cv2.imread(image_path)
    background_removed = remove_background(original)

    show(original, background_removed, "Preprocessed Image")

    image = Image.open(image_path)

    config = "-l por --oem 1 --psm 1"

    print("background_removed")
    text = pytesseract.image_to_string(background_removed, config=config)
    print(colored(text, "green"))

    print("original")
    text = pytesseract.image_to_string(preprocessed, config=config)
    print(colored(text, "green"))
