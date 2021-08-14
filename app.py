import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO
import utils
import Features
import pickle
import cv2

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "<h1>Hello World!</h1>"


@app.route('/test')
def test():
    return "<h1>Test Successful</h1>"


@app.route('/flower', methods=['POST','GET'])
def prediction():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('No file part')
            return "<h1>No File</h1>"
        file = request.files['image_file'].read()
        img = Image.open(BytesIO(file))
        img = img.convert('RGB')
        img = img.resize((224,224))
        img.save("flower.png")
        del file
        return predict(img)
    else:
        return "<h1>No File Uploaded</h1>"

@app.route('/flowerML', methods=['POST','GET'])
def predictionML():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('No file part')
            return "<h1>No File</h1>"
        file = request.files['image_file'].read()
        img = Image.open(BytesIO(file))
        img = img.convert('RGB')
        img = img.resize((224,224))
        img.save("flower.png")
        del file,img
        classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'windflower']
        with open('output/model.pkl', 'rb') as fid:
            model = pickle.load(fid)
        image = cv2.imread("flower.png")
        image= ((image+1)*255/2).astype('uint8')
        img = cv2.resize(image,(224,224))
        img = utils.process_image(img)
        cv2.imwrite("mrf.png",img)
        features = Features.extract_features(img)
        return classes[model.predict(features)[0]]
    else:
        return "<h1>No File Uploaded</h1>"
        
if __name__ == '__main__': 
    app.run(debug=True)