"""
Source: https://github.com/imanoop7/Ollama-OCR
"""

import cv2


def preprocess_image(image_path: str) -> str:
    """
    Preprocess image before OCR:
    - Auto-rotate
    - Enhance contrast
    - Reduce noise
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)

    # Auto-rotate if needed
    # TODO: Implement rotation detection and correction

    # Save preprocessed image
    preprocessed_path = f"{image_path}_preprocessed.jpg"
    cv2.imwrite(preprocessed_path, denoised)

    return preprocessed_path

if __name__ == "__main__":
    image_path = "images/3.png"
    preprocessed_path = preprocess_image(image_path)
    print(f"Preprocessed image saved at {preprocessed_path}")