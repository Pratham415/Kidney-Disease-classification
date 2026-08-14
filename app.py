import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from cnnClassifier.utils.comman import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


clApp = ClientApp()


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET'])
def trainRoute():
    os.system("dvc repro")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
def predictRoute():
    try:
        data = request.json

        if not data or 'image' not in data:
            return jsonify({"error": "No image provided in request."}), 400

        image = data['image']
        modality = data.get('modality', 'hyperspectral').lower()

        if modality not in ['hyperspectral', 'multispectral']:
            return jsonify({"error": "modality must be 'hyperspectral' or 'multispectral'"}), 400

        decodeImage(image, clApp.filename)
        classifier = PredictionPipeline(clApp.filename, modality=modality)
        result = classifier.predict()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)