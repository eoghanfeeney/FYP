import os
import pickle
import re
import cv2
import spacy

film = cv2.VideoCapture('Shrek2Movie.mp4')

with open('Normalized_Script.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    lines = text.splitlines()

# Load the English language model from spaCy
nlp = spacy.load('trained_object_model')
nlp2 = spacy.load('trained_character_model')

# Use models to process the text
doc = nlp(text)
doc2 = nlp2(text)

# Regular Expression for objects
pattern = r"\[\[(.*?)\]\]"
object_names = []

for ent in doc.ents:
    if ent.label_ == 'OBJECT_NAME' and re.match(pattern, ent.text):
        object_name = re.match(pattern, ent.text).group().rstrip(']]')
        object_name = object_name.replace("[[", "")
        if object_name.upper() not in object_names:
            object_names.append(object_name.upper())

print('Objects found from our text analysis:')
for object_name in object_names:
    print(f'{object_name}')
print()

object_dict = {name: [] for name in object_names}


for i, line in enumerate(lines):
    while '[[' in line:
        object_start = line.find('[[') + 2
        object_end = line.find(']]')
        object_str = line[object_start:object_end]
        object_str = object_str.upper()

        # if object_str not in object_names:
        #     object_names.append(object_str)

        previous_time_frame = None
        next_time_frame = None

        for j in range(i-1, -1, -1):
            if '-->' in lines[j]:
                previous_time_frame = lines[j].strip()
                start_time1, end_time1 = previous_time_frame.split(' --> ')
                break
        for j in range(i+1, len(lines)):
            if '-->' in lines[j]:
                next_time_frame = lines[j].strip()
                start_time2, end_time2 = next_time_frame.split(' --> ')
                break
        if previous_time_frame and next_time_frame:
            if object_str in object_dict:
                object_dict[object_str].append((end_time1, start_time2))
            # else:
            #     object_dict[object_str] = []
            #     object_dict[object_str].append((end_time1, start_time2))

        line = line[:object_start - 2] + line[object_end + 2:]


for obj, time_stamps in object_dict.items():
    print(f'{obj}:')
    for time_stamp in time_stamps:
        print(f'{time_stamp[0]} --> {time_stamp[1]}')
    print()


for obj, time_stamps in object_dict.items():
    for time_stamp in time_stamps:

        time_stamp = f'{time_stamp[0]} --> {time_stamp[1]}'
        start_time, end_time = time_stamp.split(' --> ')

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

        directory = f'{obj.lower()}'
        if not os.path.exists(os.path.join('frame_images', directory)):
            os.makedirs(os.path.join('frame_images', directory))

        # Frame position to the midpoint of the time stamp, *1000 because CAP_PROP_POS_MSEC reads in millisecs
        film.set(cv2.CAP_PROP_POS_MSEC, midpoint_seconds * 1000)

        # Read frame and save it as an image in the character's directory
        success, image = film.read()
        if success:
            cv2.imwrite(f'frame_images/{directory}/{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.jpg',
                        image)
        else:
            print(f'Error reading frame for {obj}')


with open('object_names.pkl', 'wb') as f:
    pickle.dump(object_names, f)

with open('object_dict.pkl', 'wb') as f:
    pickle.dump(object_dict, f)


# Regular Expression for Characters
pattern = r'([A-Z &]+):'

character_names = []

for ent in doc2.ents:
    if ent.label_ == 'CHARACTER_NAME' and re.match(pattern, ent.text):
        character_name = re.match(pattern, ent.text).group().rstrip(':')
        # character_name = re.match(pattern, ent.text).group()
        if character_name not in character_names:
            character_names.append(character_name)

print('Characters found from our text analysis:')
for character in character_names:
    print(f'{character}')
print()

# Create a dictionary to store the time stamps for each character
# Add character names extracted from text analysis into dictionary
character_dict = {name: [] for name in character_names}
all_time_stamps = []

# Regular Expression for Time stamp and characters
time_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([A-Z &]+):'

# Find matches of pattern in the text
time_matches = re.findall(time_pattern, text)

# Loop through matches and if the character name == character name found by model append timestamp
for match in time_matches:
    start_time = match[0]
    end_time = match[1]
    character_name = match[2]

    # Add the time stamp to the list of all time stamps
    time_stamp = f'{start_time[:12]} --> {end_time[:12]}'
    all_time_stamps.append(time_stamp)

    if character_name in character_names:
        character_dict[character_name].append((start_time, end_time))

    start_time, end_time = time_stamp.split(' --> ')

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

    # Set the frame position to the midpoint of the time stamp
    film.set(cv2.CAP_PROP_POS_MSEC, midpoint_seconds * 1000)

    # Check if the directory exists, if not, create it
    directory = f'{character_name.lower()}'
    if not os.path.exists(os.path.join('frame_images', directory)):
        os.makedirs(os.path.join('frame_images', directory))

    # Read the next frame and save it as an image in the character's directory
    success, image = film.read()
    if success:
        cv2.imwrite(f'frame_images/{directory}/{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.jpg', image)
    else:
        print(f'Error reading frame for {character_name}')

# Print the time stamps for each character
for character, time_stamps in character_dict.items():
    print(f'{character}:')
    for time_stamp in time_stamps:
        print(f'{time_stamp[0]} --> {time_stamp[1]}')
    print()

with open('character_dict.pkl', 'wb') as f:
    pickle.dump(character_dict, f)

with open('character_names.pkl', 'wb') as f:
    pickle.dump(character_names, f)


# # scene_pattern = r"'''[\w\s'.-]+'''"
# scene_pattern = r"'''(?:INT\.|EXT\.)\s-\s[^\n]+'''"
#

# scenes = re.split(scene_pattern, text)
#
#
# num_scenes = len(scenes)//2
# print(f'The script contains {num_scenes} scenes.')
#
# locations = []
# for i in range(1, len(scenes), 2):
#     location = scenes[i]
#     locations.append(scenes[i])
#     print(f'{location}')
