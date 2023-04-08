# import imghdr
# from PIL import Image
# import os
# import cv2
import os

#
# folder_path = "Shrek2_images/Puss"
# data_dir = 'Shrek2_images'
# image_exts = ['jpeg', 'jpg', 'bmp', 'png']
#
#
# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in image_exts:
#                 print('Image not in extension list {}'.format(image_path))
#                 os.remove(image_path)
#         except Exception as e:
#             print('Issue with Image {}'.format(image_path))
#
# for file_name in os.listdir(folder_path):
#     if file_name.endswith((".png", ".jpg", ".jpeg", ".bmp")):
#         file_path = os.path.join(folder_path, file_name)
#         with Image.open(file_path) as img:
#             iccp = img.info.get("icc_profile")
#             if iccp and b"sRGB" in iccp:
#                 print(f"{file_name} contains an incorrect sRGB profile.")

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

# Load the model
model = tf.keras.models.load_model('ImageClassificationModel')


# Evaluate the model on the TestImages data
# test_dir = 'TestImages'
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(256, 256),
#     batch_size=24,
#     class_mode='categorical')
#
# test_loss, test_acc = model.evaluate(test_generator, verbose=2)
# print('Test accuracy:', test_acc)


img_path = 'TestImages/fiona'
for image in os.listdir(img_path):

    img_width, img_height = 256, 256
    full_path = os.path.join(img_path, image)
    img = load_img(full_path, target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    img.show()

    # Make a prediction on the test image
    prediction = model.predict(x)

    # Print the predicted class
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        print('The image is Charming')
    elif predicted_class == 1:
        print('The image is Donkey')
    elif predicted_class == 2:
        print('The image is Fiona')
    elif predicted_class == 3:
        print('The image is Godmother')
    elif predicted_class == 4:
        print('The image is Puss')
    elif predicted_class == 5:
        print('The image is Shrek')
    else:
        print('Unknown character')



