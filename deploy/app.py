from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



warnings.filterwarnings("ignore")


new_model = tf.keras.models.load_model('my_model10_new.h5')

def pre(file):

        train_datagen = ImageDataGenerator(rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)


        

        training_set = train_datagen.flow_from_directory(
        './data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        color_mode='grayscale')


        # Verifing ouor Model by giving samples of cell to detect malaria
        
        test_image = image.load_img((file), target_size=(64 , 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        training_set.class_indices
       
        
        if result[0][0] == 0:
                prediction = 'NORMAL'
                
        else:
                prediction = 'PNEUMONIA'
                
        return("The Prediction Result is : "+ prediction)
        

import flask
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for , jsonify

app=flask.Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER']='imageFolder'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('testd.html')

@app.route('/index', methods=['GET'])	
def nextPage():
	# 2 Page
	return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def upload():
    if request.method == 'POST':
		# Get the file from post request
        if 'file' not in request.files:
            return  "nofile"
        else:
            file=request.files['file']
            filename = secure_filename(file.filename)
            #file.save(filename)
            predictions = pre(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify(predictions)
    else: return  "Error"
'''	
import wsgiserver
server = wsgiserver.WSGIServer(app, host='127.0.0.1', port=8081)
server.start()
'''
if __name__ == '__main__':
	app.run(debug = True)