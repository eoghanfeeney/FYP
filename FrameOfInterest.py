import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from keras.utils import load_img, img_to_array

film = cv2.VideoCapture('Shrek2Movie.mp4')
model = tf.keras.models.load_model('ImageClassificationModel3')

# Frame count
total_frames = int(film.get(cv2.CAP_PROP_FRAME_COUNT))

image_index = 0


def image_dataset(directory, character_object, film):
    root = tk.Tk()
    root.title(f"Frames of Interest of {character_object}")
    root.configure(bg='#90EE90')
    root.geometry("1000x1000")

    # get image paths
    image_paths = []
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
    image_paths = sorted(image_paths)

    # Display Image count
    image_count = tk.Label(root, text=f"Image 1 of {len(image_paths)} of {character_object}")
    image_count.pack()
    image_count.config(bg='#90EE90')

    # Create canvas
    canvas = tk.Canvas(root, width=850, height=850)
    canvas.pack()

    image_name_label = tk.Label(root, text="")
    image_name_label.pack()

    # Show current image
    def show_image(display_image_index, film):
        global image_index

        image = Image.open(image_paths[image_index])
        image = image.resize((850, 850), Image.ANTIALIAS)

        # convert to display on canvas
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo
        image_count.config(text=f"Image {display_image_index + 1} of {len(image_paths)} of {character_object}")

        image_name = os.path.basename(image_paths[image_index])
        image_name_label.config(text=image_name.replace("-", ":").replace("_", " --> ").strip(".jpg"), bg='#90EE90')

        time = image_name_label.cget("text").strip('.jpg')
        start_time, end_time = time.split(' --> ')

        # Convert the time to seconds
        time_multiplier = [3600, 60, 1, 0.001]

        start_time_parts = start_time.replace(',', '.').replace(':', '.').split('.')
        start_seconds = 0

        for i in range(len(start_time_parts)):
            time_unit = float(start_time_parts[i])
            start_seconds += time_unit * time_multiplier[i]

        end_time_parts = end_time.replace(',', '.').replace(':', '.').split('.')
        end_seconds = 0

        for i in range(len(end_time_parts)):
            time_unit = float(end_time_parts[i])
            end_seconds += time_unit * time_multiplier[i]

        midpoint_seconds = (start_seconds + end_seconds) / 2

        # Set the frame position to image
        film.set(cv2.CAP_PROP_POS_MSEC, midpoint_seconds * 1000)

    # show next image
    def show_next_image():
        global image_index

        if image_index < len(image_paths) - 1:
            image_index += 1
            show_image(image_index, film)
        else:
            image_index = 0
            show_image(image_index, film)

    # show previous image
    def show_previous_image():
        global image_index

        if image_index > 0:
            image_index -= 1
            show_image(image_index, film)
        else:
            image_index = len(image_paths) - 1
            show_image(image_index, film)

    # show next Frame
    def next_frame(film, skip):

        for i in range(skip):
            success, frame = film.read()
            if not success:
                # no frames left
                image_name_label.config(text="No Frames Left")
                return

        # read the ith frame
        success, frame = film.read()

        if success:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            image = image.resize((850, 850), Image.ANTIALIAS)

            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo

        else:
            image_name_label.config(text="No Frames Left")

    def prev_frame(film, skip):

        frame_pos = film.get(cv2.CAP_PROP_POS_FRAMES)
        frame_pos = max(frame_pos - 1 - skip, 0)

        film.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        success, frame = film.read()
        if success:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(image)
            image = image.resize((850, 850), Image.ANTIALIAS)

            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.image = photo

        else:
            image_name_label.config(text="No Frames Left")

    def image_class():
        image_name = image_paths[image_index]
        print(image_name)
        img_width, img_height = 256, 256
        img = load_img(image_name, target_size=(img_width, img_height))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # Make a prediction on the test image
        prediction = model.predict(x)

        # Print the predicted class
        predicted_class = np.argmax(prediction)
        print(predicted_class)

        if predicted_class == 0:
            print('The image is Charming')
            result_label.config(text="The image is Charming")
        elif predicted_class == 1:
            print('The image is Donkey')
            result_label.config(text="The image is Donkey")
        elif predicted_class == 2:
            print('The image is Fiona')
            result_label.config(text="The image is Fiona")
        elif predicted_class == 3:
            print('The image is Godmother')
            result_label.config(text="The image is Godmother")
        elif predicted_class == 4:
            print('The image is Puss')
            result_label.config(text="The image is Puss")
        elif predicted_class == 5:
            print('The image is Shrek')
            result_label.config(text="The image is Shrek")
        else:
            print('Unknown character')
            result_label.config(text="Unknown Character")

    # create buttons
    previous_button = tk.Button(root, text="Previous Image", command=show_previous_image)
    previous_button.pack(side=tk.LEFT)
    previous_button.config(bg='#90EE90')
    next_button = tk.Button(root, text="Next Image", command=show_next_image)
    next_button.pack(side=tk.RIGHT)
    next_button.config(bg='#90EE90')

    next_50_frame_button = tk.Button(root, text="View 50th Next Frame", command=lambda: next_frame(film, 49))
    next_50_frame_button.pack(side=tk.RIGHT)
    next_50_frame_button.config(bg='#90EE90')
    prev_50_frame_button = tk.Button(root, text="View 50th Previous Frame", command=lambda: prev_frame(film, 50))
    prev_50_frame_button.pack(side=tk.LEFT)
    prev_50_frame_button.config(bg='#90EE90')

    next_frame_button = tk.Button(root, text="View Next Frame", command=lambda: next_frame(film, 0))
    next_frame_button.pack(side=tk.RIGHT)
    next_frame_button.config(bg='#90EE90')
    prev_frame_button = tk.Button(root, text="View Previous Frame", command=lambda: prev_frame(film, 1))
    prev_frame_button.pack(side=tk.LEFT)
    prev_frame_button.config(bg='#90EE90')

    msg = tk.Label(root, text="Try Changing Frames!")
    msg.pack()
    msg.config(bg='#90EE90')

    if character_object in ['SHREK', 'DONKEY', 'FIONA', 'PUSS', 'CHARMING', 'GODMOTHER']:
        image_class_button = tk.Button(root, text="Image Classification", command=lambda: image_class())
        image_class_button.pack()
        image_class_button.config(bg='#90EE90')

    result_label = tk.Label(root, text="", bg='#90EE90')
    result_label.pack()


    # show first image
    show_image(image_index, film)

    response = input(f"Found {len(image_paths)} images in {directory}. Display images? (Y/N): ")
    if response.lower() == "y" or response.lower() == "yes":
        root.mainloop()
    else:
        root.destroy()


# Release the video capture object
film.release()

