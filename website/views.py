import pickle
import os

import numpy as np
from PIL.Image import Image
from flask import Blueprint, render_template, send_from_directory
import tensorflow as tf
from keras.utils import load_img, img_to_array

views = Blueprint('views', __name__)
model = tf.keras.models.load_model('ImageClassificationModel3')


@views.route('/')
def home():
    return render_template("home.html")


@views.route('/characters')
def characters():
    with open('character_names.pkl', 'rb') as f:
        character_names = pickle.load(f)

    with open('character_dict.pkl', 'rb') as f:
        character_dict = pickle.load(f)
    return render_template('characters.html', character_names=character_names, character_dict=character_dict)


@views.route('/displaycharacter/<character>')
def display_characters(character):
    character = character.lower()
    directory_path = f'frame_images/{character}'

    image_files = os.listdir(directory_path)

    for image_file in image_files:
        image_path = f'{directory_path}/{image_file}'
        print(image_path)

    predictions = []
    img_width, img_height = 256, 256
    for image_file in image_files:
        image_path = f'{directory_path}/{image_file}'
        img = load_img(image_path, target_size=(img_width, img_height))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        prediction = model.predict(x)
        predicted_class = int(np.argmax(prediction))
        predictions.append(predicted_class)
    print(predictions)
    # Pass the image file dictionary
    return render_template("displaycharacter.html", character=character, image_files=image_files, predictions=predictions)


# @views.route("/frame_images/<character>/<path:filename>")
# def serve_image(character, filename):
#     # Define the path to the requested image file
#     character = character.lower()
#     directory_path = f'frame_images/{character}'
#
#     print(filename)
#     print(directory_path)
#     # Return the image file as binary data
#     return send_from_directory(directory_path, filename)

@views.route('/objects')
def objects():
    with open('object_names.pkl', 'rb') as f:
        object_names = pickle.load(f)

    with open('object_dict.pkl', 'rb') as f:
        object_dict = pickle.load(f)
    return render_template('objects.html', object_names=object_names, object_dict=object_dict)


@views.route('/displayobject/<object>')
def display_objects(object):
    object = object.lower()
    directory_path = f'frame_images/{object}'

    # Get a list of all the image file names in the directory
    image_files = os.listdir(directory_path)

    for image_file in image_files:
        image_path = f'{directory_path}/{image_file}'
        print(image_path)

    # Pass the dictionary of image files to the template
    return render_template("displayobject.html", object=object, image_files=image_files)

