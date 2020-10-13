import os
import keras
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model_age=keras.models.load_model("files/age.h5")
model_gender=keras.models.load_model("files/gender.h5")
print("Model Loaded")

def predict(img_file,img_shape=(64,64)):
    print(type(img_file))
    label_age=["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-100"]
    label_gender=["Female","Male"]
    img = img_to_array(img_file)
    print('image loaded')
    img=img.reshape(-1,img_shape[0],img_shape[1],1)
    print('Predicting')
    pred_age=model_age.predict(img)
    pred_gender=model_gender.predict(img)
    return label_age[np.argmax(pred_age)]+ " Y, "+label_gender[np.argmax(pred_gender)]

@app.route('/')
def hello_world():
    return "<h1>Hello World!</h1>"

@app.route('/test')
def test():
    return "<h1>Test Successful</h1>"

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    print("Processing Started")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image_file' not in request.files:
            flash('No file part')
            return "<h1>No File</h1>"
        file = request.files['image_file'].read()
        img = Image.open(BytesIO(file)).convert('LA')
        print(img.size)
        img = img.resize((64,64), Image.ANTIALIAS)
        print(img.size)
        return predict(img)
    else:
        return "<h1>No File Uploaded</h1>"

if __name__ == '__main__':
    app.run(debug=True)