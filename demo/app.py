import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# PyTorch
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms

# TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras

# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Some utilites
import hashlib
import numpy as np
from util import base64_to_pil

# Local imports
sys.path.append('..')
from src.sample import build_model_and_vocab, sample
from compiler.classes.Compiler import *
from compiler.compile import compile


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# # Model saved with Keras model.save()
# MODEL_PATH = 'models/your_model.h5'

# # Load your own trained model
# # model = load_model(MODEL_PATH)
# # model._make_predict_function()          # Necessary
# # print('Model loaded. Start serving...')


# def model_predict(img, model):
#     img = img.resize((224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='tf')

#     preds = model.predict(x)
#     return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/convert', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        model, vocab = build_model_and_vocab('../bootstrap4.vocab', '../weights/best.pkl')
        prediction = sample(img, model, vocab)

        content_hash = prediction.replace(" ", "").replace("\n", "")
        content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

        gui_file_path = os.path.join('static/output', content_hash + '.gui')

        with open(gui_file_path, 'w') as f:
            f.write(prediction)

        output_file_path = compile(Compiler('../compiler/assets/bootstrap-config.json'), gui_file_path, r'\[\]')

        # Serialize the result, you can add additional fields
        return jsonify(result=output_file_path)

        # Make prediction
        # preds = model_predict(img, model)

        # # Process your result for human
        # pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # # Serialize the result, you can add additional fields
        # return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
