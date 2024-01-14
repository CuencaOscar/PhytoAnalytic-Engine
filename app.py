from flask import Flask, request, jsonify
from PIL import Image
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'


# TODO Debo adaptar este codigo para
# recibir imagen y predecir (3 Respuestas)
@app.route('/api', methods=['POST'])
def api():
    print('Veamos si entra esto')

    content = request.form.get('image')
    f = open('image.jpg', 'wb')
    response = requests.get(content)
    f.write(response.content)
    f.close()
    img = Image.open("image.jpg")
    img.save("image.jpg", optimize=True, quality=75)

    return jsonify({"prediction": 'result'})


@app.route('/')
def init():
    return "Tesis Prediction"


if __name__ == '__main__':
    app.run(debug=True)
