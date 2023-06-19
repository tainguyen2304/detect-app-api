
from PIL import Image
from flask import *
from ultralytics import YOLO
import os
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

modelPerson = YOLO('yolov8x.pt')
modelFace = YOLO('yolov8n-face.pt')
img = os.path.join('')


@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    image_url = Image.open(image)
    result_ps = modelPerson.predict(source=image_url)
    result_face = modelFace.predict(source=image_url)
    dict_classes_modelPerson = modelPerson.model.names
    dict_classes_modelFace = modelFace.model.names
    classes_modelPerson = result_ps[0].boxes.cls.cpu().numpy()
    classes_modelFace = result_face[0].boxes.cls.cpu().numpy()
    labels_ps = [dict_classes_modelPerson[i] for i in classes_modelPerson]
    labels_face = [dict_classes_modelFace[i] for i in classes_modelFace]
    persons = 0
    for label in labels_ps:
        if label == 'person':
            persons += 1
    matrixImg = []
    if persons > len(labels_face):
        matrixImg = result_ps[0].plot()
    else:
        matrixImg = result_face[0].plot()
    img = Image.fromarray(matrixImg)
    img.save('result.png')
    resp = jsonify({
        "message": 'Files success upload',
        "status": 'success',
        "data": {
            "labelsPersonModel": labels_ps,
            "labelsFaceModel": labels_face,
            "imgResult": "http://192.168.11.31:8080/result"
        }
    })

    resp.status_code = 200
    print('1')
    return resp


@app.route('/result')
def home():
    print('2')
    file = os.path.join(img, 'result.png')
    return send_file(file, mimetype='image/png')


port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
