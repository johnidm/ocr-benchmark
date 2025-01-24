# OCR Project

This project provides various OCR (Optical Character Recognition) implementations using different libraries and APIs. The goal is to extract text from images using multiple OCR engines and compare their performance.

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

To install Tesseract, run:

```sh
apt install -y libtesseract-dev tesseract-ocr tesseract-ocr-eng tesseract-ocr-por
```

## Usage

The `images` folder includes some exampled of images used for OCR. 

### Benchmarking OCR Engines

The  `benchmark.py` script benchmarks different OCR engines. You can run it to see the performance of each engine on a sample image.

```sh
python benchmark.py
```

### Google Vision API

The `GoogleVisionAPI.ipynb` notebook demonstrates how to use the Google Vision API for OCR. Open the notebook in Jupyter and follow the instructions to analyze images.

### Visualizing Images

The `VisualizeImage.ipynb` notebook provides utilities to visualize images and their processed versions. Open the notebook in Jupyter to see the visualizations.

### Pre-processing Images

The `pre_processing.py` script contains functions to pre-process images before applying OCR. This includes converting images to grayscale, thresholding, and removing backgrounds.

### Surya OCR

The `surya_ocr.py` script demonstrates how to use the Surya OCR library for text extraction.

### DocTR

The `doctr_kie.py` and `doctr_ocr.py` scripts demonstrates how to use the differents DocTR models for text extraction .

## OCR Engines

The project includes implementations for the following tool to use for OCR:

### Generative AI

- [OpenAI](https://openai.com/)
- [Gemini](https://deepmind.google/technologies/gemini/)

### Vision AI

- [Google Vision AI](https://cloud.google.com/vision?hl=en)
- [Azure Vision AI](https://azure.microsoft.com/en-us/products/ai-services/ai-vision)

### Open Source Libraries

- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [DocTR](https://github.com/mindee/doctr)
- [Surya OCR](https://github.com/VikParuchuri/surya)
