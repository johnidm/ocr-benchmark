from PIL import Image
from surya.model.detection.model import (
    load_model as load_det_model,
)
from surya.model.detection.model import (
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr  # pip install surya-ocr
from termcolor import colored


def to_text(image_path: str) -> str:
    image = Image.open(image_path)

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

    return " ".join(text)

images_path = "images/1.jpg"

text = to_text(images_path)
print(colored(text, "green"))
