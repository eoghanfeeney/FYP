import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import load_img, img_to_array
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

# Image data directory
data_path = 'Shrek2_images'

# set image height and width
img_width = 256
img_height = 256

# Create Data Generator
datagen = ImageDataGenerator(
    validation_split=0.3,
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest')

# training data
generator = datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=28,
    class_mode='categorical',
    subset='training')

# Validation data
val_generator = datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=28,
    class_mode='categorical',
    subset='validation')

# The CNN
model = Sequential()

model.add(Conv2D(16, (3, 3), 1,  activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2, 2)))

# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model
history = model.fit(
    generator,
    epochs=20,
    validation_data=val_generator)


model.save('ImageClassificationModel3')


# Plot the training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()




