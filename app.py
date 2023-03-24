from flask import Flask, render_template, request
import tensorflow as tf
from get_features import get_features, extact_features
from keras.utils import load_img
import pickle
from PIL import Image
import os
import numpy as np
from check import check
model = tf.keras.models.load_model('models/model')
with open('./models/classes.pkl', 'rb') as pickle_load:
    classes = pickle.load(pickle_load)
app = Flask(__name__, static_url_path='/static')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/upload')
def upload():
    return render_template('predict.html')

@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        x = Image.open(request.files['image'])
        img = x.convert('L')
        img.save("uploads/image.jpg")
        img_path = os.path.join(os.path.dirname(__file__), 'uploads/image.jpg')
        x = x.convert("RGB")
        x.save("uploads/image2.jpg")
        img_path2 = os.path.join(os.path.dirname(__file__), 'uploads/image2.jpg')
        os.path.isfile(img_path)
        if check(img_path2):
            img_g = load_img(img_path,target_size = (331,331,3))
            img_g = np.expand_dims(img_g, axis=0)
            test_features = extact_features(img_g)
            predg = model.predict(test_features)
            print(np.max(predg[0]))
            return render_template('classify.html', breed = classes[np.argmax(predg[0])])
        else:
            return render_template('predict.html',message="No dog found!!")
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run()