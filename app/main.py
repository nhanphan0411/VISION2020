# from blueprints import home_page
from flask import Flask, Blueprint, jsonify, render_template, request
import tensorflow as tf
import re
import base64
import uuid
import numpy as np 
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2

app = Flask(__name__)
# app.register_blueprint(home_page)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/static/images/'
app.config['OUTPUT_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'output')

model = tf.keras.models.load_model('/Volumes/_akai_/SRGANS/flask_app/app/model/gen_model_2000(f2k_2202).h5')

def parse_image(imgData):
    img_str = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.decodebytes(img_str)
    filename = "{}.jpg".format(uuid.uuid4().hex)
    with open('uploads/'+filename, "wb") as f:
        f.write(img_decode)
    return img_decode

def preprocess(image):
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.
    image = tf.expand_dims(image, 0)
    return image

def image_to_b64(array):
    array = array[:,:,::-1]
    flag, arr = cv2.imencode('.png', array)
    res = arr.tobytes()
    res = base64.b64encode(res)
    return res.decode("UTF-8")


@app.route('/', methods = ['GET', 'POST'])
@app.route('/home', methods = ['GET', 'POST'])
def index():
    return render_template('home_page.html') 

@app.route('/upload', methods = ['POST'])
def upload():
    data = request.data
    image = parse_image(data)
    image = preprocess(image)

    prediction = model.predict(image)
    print(prediction.shape)
    prediction = (prediction[0]*255).astype(int)
    # plt.imsave('predict.png', prediction)
    # prediction.tobytes()
    b64 = image_to_b64(prediction)
    print(str(b64)[:10])
    return jsonify(b64)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000, debug=True)
