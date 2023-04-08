import os
import random
import string

directory = 'C:/Users/Eoghan/Downloads/charming2'

image_files = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]


def generate_random_string():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))


for image_file in image_files:
    new_file_name = generate_random_string() + os.path.splitext(image_file)[1]

    os.rename(os.path.join(directory, image_file), os.path.join(directory, new_file_name))
