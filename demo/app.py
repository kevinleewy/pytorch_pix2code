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

# model_path = '../weights/checkpoint-10-1.pkl'
# model_path = '../weights/checkpoint-20-1.pkl'
# model_path = '../weights/checkpoint-30-1.pkl'
# model_path = '../weights/checkpoint-40-1.pkl'
model_path = '../weights/best.pkl'
model, vocab = build_model_and_vocab('../bootstrap4.vocab', model_path)

beam_size = 10

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

        prediction = sample(img, model, vocab, beam_size)

        content_hash = prediction.replace(" ", "").replace("\n", "")
        content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

        gui_file_path = os.path.join('static/output', content_hash + '.gui')

        with open(gui_file_path, 'w') as f:
            f.write(prediction)

        output_file_path = compile(Compiler('../compiler/assets/bootstrap-config.json'), gui_file_path, r'\[\]')

        # Serialize the result, you can add additional fields
        return jsonify(result=output_file_path)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
