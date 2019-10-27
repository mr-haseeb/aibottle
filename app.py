from __future__ import division, print_function
import os
import sys
import os
import glob
import re

import tensorflow as tf
# from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_ex-024_acc-0.996875.h5'


model=tf.keras.models.load_model(MODEL_PATH)

model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')



def model_predict(file):
    # img_rgb = cv2.resize(img_rgb,(224,224),3)  # resize
    img_array = cv2.imread(file)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb,(224,224),3) 
    img_rgb = np.array(img_rgb).astype(np.float32)/255.0  # scaling
    img_rgb = np.expand_dims(img_rgb, axis=0)
    array = model.predict(img_rgb)
    result = array[0]
    answer = np.argmax(result)

    preds = model.predict(img_rgb)
    return preds

    # # x = load_img(file, target_size=(img_width,img_height))
    # # x = img_to_array(x)
    # # x = np.expand_dims(x, axis=0)
    # # array = model.predict(x)
    # # result = array[0]
    # # answer = np.argmax(result)
    # if answer == 0:
    #     print("Not Predicted")
    # elif answer == 1:
	#     print("Predicted ")
    # else:
    #     print("NOT Sure")
    # return answer

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        array  = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
           # ImageNet Decode
        result = array[0]
        answer = np.argmax(result)
        if answer == 0:
            result="Not Solan de Cabras"
        elif answer == 1:
            result="Solan de Cabras"
        else:
            result="Not Sure"        # Convert to string
        
        return result
    return None


if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)