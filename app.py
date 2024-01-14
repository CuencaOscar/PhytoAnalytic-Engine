from flask import Flask, request
import requests
import torchvision.transforms as transforms
from Classifier import Classifier
from torchvision.datasets import ImageFolder
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


@app.route('/api', methods=['POST'])
def api():

    content = request.form.get('image')

    output_directory = "predict_image/test"

    os.makedirs(output_directory, exist_ok=True)

    image_path = os.path.join(output_directory, 'image.jpg')

    response = requests.get(content)

    with open(image_path, 'wb') as f:
        f.write(response.content)

    test = ImageFolder('./predict_image', transform=transforms.ToTensor())
    img, label = test[0]
    predictions = Classifier.predict_image(img, Classifier.model)
    prediction_data = []
    for pred_class, pred_prob in predictions:
        prediction_data.append({
            "class": pred_class,
            "probability": "{:.2f}".format(pred_prob)
        })
    return prediction_data


@app.route('/')
def init():
    return "Tesis Prediction"


if __name__ == '__main__':
    app.run(debug=True)
