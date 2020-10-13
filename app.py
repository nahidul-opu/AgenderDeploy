import os
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model_age=None
model_gender=None

label_age=["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-100"]
label_gender=["Female","Male"]

def predict(img_file,img_shape=(64,64)):
    img = img_to_array(img_file)
    img=img.reshape(-1,img_shape[0],img_shape[1],1)
    global model_age
    if model_age is None:   
        model_age=load_model("files/age.h5")
    global model_gender    
    if model_gender is None:
        model_gender=load_model("files/gender.h5")
    pred_age=model_age.predict(img)
    pred_gender=model_gender.predict(img)
    result= label_age[np.argmax(pred_age)]+ " Y, "+label_gender[np.argmax(pred_gender)]
    del pred_age,pred_gender,img_file,img,img_shape
    return result


@app.route('/')
def hello_world():
    return "<h1>Hello World!</h1>"


@app.route('/test')
def test():
    return "<h1>Test Successful</h1>"


@app.route('/prediction', methods=['POST','GET'])
def prediction():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            flash('No file part')
            return "<h1>No File</h1>"
        file = request.files['image_file'].read()
        img = Image.open(BytesIO(file)).convert('LA')
        del file
        img = img.resize((64,64), Image.ANTIALIAS)
        return predict(img)
    else:
        return "<h1>No File Uploaded</h1>"

if __name__ == '__main__': 
    app.run(debug=True)