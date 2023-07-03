from tkinter import Image
import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, render_template, flash, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
MODEL_PATH = 'besttransfermodel.h5'
# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')
# C:\Users\Admin\PycharmProjects\flask
def preprossing(image):
    image = Image.open(image)
    image = image.resize((240, 240))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 240, 240, 3)
    return image_arr
def model_predict(img_path, model):
    img = image.load_img(img_path,target_size=(240, 240))  # target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255

    preds = model.predict(img)
    print(preds)
    pred = np.floor(preds)
    print(pred)
    return pred
#classes = ['does not have a tumor', 'have a tumor']
#model = load_model("besttransfermodel.h5")
@app.route('/')
def index():
    return render_template('index.html', appName="Brain Tumor")
@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files['fileup']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)
        str0 = 'No Tumor'
        str1 = 'Tumor'
        str3 = 'pituitary'
        str2 = 'No aaaaaTumour'
        print(pred)
        if pred[0] == 0:
            prediction = 'No Tumor'
        elif pred[0] == 1:
            prediction = 'Tumor'

        elif pred[0] == 3:
            prediction = 'pituitary'
        else:
            prediction = 'No aaaaaTumour'
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

         #Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)
        str0 = 'No Tumor'
        str1 = 'Tumor'
        str3 = 'pituitary'
        str2 = 'No aaaaaTumour'
        print(pred)
        if pred[0] == 0:
            prediction='No Tumor'
        elif pred[0] == 1:
            prediction='Tumor'

        elif pred[0] == 3:
            prediction='pituitary'
        else:
            prediction='No aaaaaTumour'
    #return None

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Brain Tumor")
    else:
        return render_template('index.html', appName="Brain Tumor")






























if __name__ == '__main__':
    app.run(debug=True)
