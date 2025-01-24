import base64
import json
import re
from base64 import b64encode

import cv2
import easyocr
import google.generativeai as genai # https://cloud.google.com/vision/docs/drag-and-drop?hl=en
import pytesseract
import requests
from azure.ai.vision.imageanalysis import ImageAnalysisClient # https://portal.vision.cognitive.azure.com/demo/extract-text-from-images
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from doctr.io import DocumentFile
from doctr.models import kie_predictor, ocr_predictor
from openai import OpenAI
from PIL import Image
from surya.model.detection.model import (
    load_model as load_det_model,
)
from surya.model.detection.model import (
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr
from termcolor import colored

from ocr_pre_processing import remove_background

OPENAI_API_KEY = "<insert here your OpenAI API Key>"
GOOGLE_VISION_API_KEY = "<insert here your Google Vision API Key>"
GEMINI_API_KEY = "<insert here your Gemini API Key>"
AZURE_VISION_API_KEY = "<insert here your Azure Vision API Key>"


genai.configure(api_key=GEMINI_API_KEY)


class Gemini:
    def to_text(self, image_path):
        sample_file = Image.open(image_path)

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = "OCR this image. Do not include any markdown or code formatting."
        response = model.generate_content([prompt, sample_file])

        return response.text


class AzureVisionAI:
    def to_text(self, image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()

        region = "eastus"
        endpoint = "https://ocrapirest.cognitiveservices.azure.com/"

        client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(AZURE_VISION_API_KEY),
            region=region,
        )

        response = client.analyze(
            image_data,
            visual_features=[VisualFeatures.READ],
        )

        lines = []

        if response.read is not None:
            for line in response.read.blocks[0].lines:
                lines.append(line.text)

        return "\n".join(lines)


class GoogleVisionAI:
    def to_text(self, image_path):
        with open(image_path, "rb") as f:
            content = b64encode(f.read()).decode()

            payload = {
                "image": {
                    "content": content,
                },
                "features": [
                    {
                        "type": "DOCUMENT_TEXT_DETECTION",
                        "maxResults": 50,
                    }
                ],
            }

        data = json.dumps({"requests": payload}).encode()

        response = requests.post(
            url="https://vision.googleapis.com/v1/images:annotate",
            data=data,
            params={
                "key": GOOGLE_VISION_API_KEY,
            },
            headers={
                "Content-Type": "application/json",
            },
        )

        if not response.ok:
            raise Exception(
                f"Status Code: {response.status_code} - Reason {response.text}"
            )

        results = response.json()["responses"][0]

        return results["fullTextAnnotation"]["text"]


class OpenAIOCR:
    def encode_image(self, image_path):
        """
        Encodes an image file to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def to_text(self, filename):
        client = OpenAI(api_key=OPENAI_API_KEY)

        base64_image = self.encode_image(filename)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "OCR this image. Do not include any markdown or code formatting.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


class Surya:
    def to_text(self, filename: str) -> str:
        image = Image.open(filename)

        langs = ["pt"]
        det_processor, det_model = load_det_processor(), load_det_model()
        rec_model, rec_processor = load_rec_model(), load_rec_processor()

        text = []

        predictions = run_ocr(
            [image], [langs], det_model, det_processor, rec_model, rec_processor
        )
        for prediction in predictions:
            for line in prediction.text_lines:
                text.append(line.text)

        return "\n".join(text)


class Tesseract:
    def to_text(self, filename: str) -> str:
        image = remove_background(cv2.imread(filename))

        config = "-l por --oem 1 --psm 1"
        text = pytesseract.image_to_string(image, config=config)

        return text


class EasyOCR:
    def to_text(self, filename: str) -> str:
        texts = []
        image = cv2.imread(filename)
        reader = easyocr.Reader(["pt"])
        results = reader.readtext(image)
        for _, text, _ in results:
            texts.append(text)

        return "\n".join(texts)


class DocOCR:
    def to_text(self, filename: str) -> str:
        model = ocr_predictor("db_resnet50", "sar_resnet31", pretrained=True)

        doc = DocumentFile.from_images(filename)
        result = model(doc)
        page = result.pages[0]

        words = []
        for block in page.blocks:
            for line in block.lines:
                texts = [word.value for word in line.words]
                words.append(" ".join(texts))

        return " ".join(words)


class DocOCRKIE:
    def to_text(self, filename: str) -> str:
        model = kie_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        )
        doc = DocumentFile.from_images(filename)
        result = model(doc)

        texts = []
        predictions = result.pages[0].predictions
        for class_name in predictions.keys():
            list_predictions = predictions[class_name]
            for prediction in list_predictions:
                texts.append(prediction.value)

        return " ".join(texts)


def norm(value):
    return " ".join(value.split())


def show(name, text):
    print("--------------------------")
    print(colored(name, "light_green", attrs=["bold"]))
    print(colored(norm(text), "light_green"))


def search(text, pattern):
    result = re.search(pattern, text)
    if result:
        return result.group()
    return ""


def preprocess(text):
    text = re.sub('N"', "N°", text)
    return text


def findall(text, pattern):
    return re.findall(pattern, text)


def regex(text):
    print(colored("Regex Results", "light_green"))

    regex_01 = search(text, r"F.*?H")
    regex_02 = findall(text, r"\b\d{20}\b")
    regex_03 = findall(text, r"N°\.?\s?([\d.]+)")

    print(colored(f"   01: {regex_01}", "light_green"))
    print(colored(f"   02: Total: {len(regex_02)} - {regex_02}", "light_green"))
    print(colored(f"   03: {regex_03}", "light_green"))
    print(colored("---------------------------", "light_green"))


image_path = "images/1.jpg"


engines = [
    # Generative AI
    ("OpenAI", OpenAIOCR()),
    ("Gemini", Gemini()),
    # Bibliotecas
    ("DocOCR KIE", DocOCRKIE()),
    ("DocOCR", DocOCR()),
    ("Surya", Surya()),
    ("EasyOCR", EasyOCR()),
    ("Tesseract", Tesseract()),
    # Vision AI
    ("Google Vision AI", GoogleVisionAI()),
    ("Azure Vision AI", AzureVisionAI()),
]


results = []

for name, engine in engines:
    text = engine.to_text(image_path)
    results.append((name, text))


for name, text in results:
    show(name, text)
    regex(text)
