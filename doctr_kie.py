import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import kie_predictor
from termcolor import colored


def execute_model(model, image_path, show=False):
    doc = DocumentFile.from_images(image_path)

    result = model(doc)
    if show:
        result.show()
        synthetic_pages = result.synthesize()
        plt.imshow(synthetic_pages[0])
        plt.axis("off")
        plt.show()

    predictions = result.pages[0].predictions

    texts = []
    predictions = result.pages[0].predictions
    for class_name in predictions.keys():
        list_predictions = predictions[class_name]
        for prediction in list_predictions:
            texts.append(prediction.value)

    return " ".join(texts)


models = [
    (
        "db_resnet50+crnn_vgg16_bn",
        kie_predictor(
            det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
        ),
    ),
    (
        "db_resnet50+sar_resnet31",
        kie_predictor(
            det_arch="db_resnet50", reco_arch="sar_resnet31", pretrained=True
        ),
    ),
]

images_path = "images/1.jpg"

for name, model in models:
    words = execute_model(model, images_path, True)
    print(colored(f"{name}", "green", attrs=["bold"]))
    print(colored(words, "green"))
