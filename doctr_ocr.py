import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from termcolor import colored


def execute_model(images_path, model, show=False):
    doc = DocumentFile.from_images(images_path)

    result = model(doc)
    if show:
        result.show()
        synthetic_pages = result.synthesize()
        plt.imshow(synthetic_pages[0])
        plt.axis("off")
        plt.show()

    predictions = result.export()
    pages = predictions["pages"][0]
    blocks = pages["blocks"]

    words = []
    for b in blocks:
        for line in b["lines"]:
            texts = [w["value"] for w in line["words"]]
            words.append(" ".join(texts))
    return words


models = [
    (
        "default",
        ocr_predictor(
            pretrained=True,
        ),
    ),
    (
        "db_resnet50+crnn_vgg16_bn",
        ocr_predictor(
            "db_resnet50",
            "crnn_vgg16_bn",
            pretrained=True,
        ),
    ),
    (
        "linknet_resnet18",
        ocr_predictor(
            "linknet_resnet18",
            pretrained=True,
            assume_straight_pages=False,
            preserve_aspect_ratio=True,
        ),
    ),
    (
        "db_resnet50+crnn_mobilenet_v3_large",
        ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=False,
            preserve_aspect_ratio=True,
        ),
    ),
    (
        "batch size",
        ocr_predictor(
            pretrained=True,
            det_bs=4,
            reco_bs=1024,
        ),
    ),
    (
        "db_resnet50+sar_resnet31",
        ocr_predictor(
            "db_resnet50",
            "sar_resnet31",
            pretrained=True,
        ),
    ),
    (
        "db_resnet50+vitstr_base",
        ocr_predictor(
            "db_resnet50",
            "vitstr_base",
            pretrained=True,
        ),
    ),
]

images_path = "images/1.jpg"

results = []

for name, model in models:
    words = execute_model(images_path, model)
    line = " ".join(words)
    results.append((name, line))

for name, line in results:
    print(colored(f"{name}", "green", attrs=["bold"]))
    print(colored(line, "green"))
