import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import keras

UPLOAD_FOLDER = 'files'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict(image_path,grayscale=True,img_shape=(64,64),channel=1):
    print(image_path)
    label_age=["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-100"]
    label_gender=["Female","Male"]
    img=cv2.imread(image_path)
    if grayscale:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('image loaded')
    img=cv2.resize(img,img_shape)
    print('resized')
    img=img.reshape(-1,img_shape[0],img_shape[1],channel)
    print('Predicting')
    global model_age
    global model_gender
    try:
        pred_age=model_age.predict(img)
        pred_gender=model_gender.predict(img)
        return label_age[np.argmax(pred_age)]+ " Y, "+label_gender[np.argmax(pred_gender)]
    except:
        return "Failed"

@app.route('/')
def hello_world():
    return 'Hello worlds!'

@app.route('/test')
def test():
    return 'Allah help me'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    print("Processing Started")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image_file' not in request.files:
            flash('No file part')
            return "No File"
        file = request.files['image_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "Empty File"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
            except:
                return "Upload Failed"
            return predict(image_path)
        else:
            return "Invalid File"
    else:
        return "No File Uploaded"

model_name_age=os.path.join(app.config['UPLOAD_FOLDER'], "age.h5")
print('loading age model from: ' + model_name_age)
model_age=keras.models.load_model(model_name_age)

model_name_gender=os.path.join(app.config['UPLOAD_FOLDER'], "gender.h5")
print('loading model from: ' + model_name_gender)
model_gender=keras.models.load_model(model_name_gender)
print("Model Loaded")

if __name__ == '__main__':
    app.run()
'''
import sys


sys.path.insert(0, os.path.dirname(__file__))


def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    message = 'It works!\n'
    version = 'Python v' + sys.version.split()[0] + '\n'
    response = '\n'.join([message, version])
    return [response.encode()]'''