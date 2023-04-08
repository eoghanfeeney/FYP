import pickle
import cv2
from FrameOfInterest import image_dataset

film = cv2.VideoCapture('Shrek2Movie.mp4')

with open('character_names.pkl', 'rb') as f:
    character_names = pickle.load(f)

with open('character_dict.pkl', 'rb') as f:
    character_dict = pickle.load(f)

with open('object_names.pkl', 'rb') as f:
    object_names = pickle.load(f)

with open('object_dict.pkl', 'rb') as f:
    object_dict = pickle.load(f)


while True:
    char_or_object = input('Would you like to see the characters or objects our text analysis has found? (C/O) ')

    if char_or_object.upper() == 'C':
        print('Characters found from our text analysis:')
        for character in character_names:
            print(f'{character}')
        print()

        while True:
            character = input("Enter the character you would like to display: ")
            print()
            character_info = character.upper()

            frame_count = 0
            time_stamp_count = 0

            if character_info in character_dict:
                print(f'Time stamps for {character}:')
                for time_stamp in character_dict[character_info]:
                    time_stamp_count += 1
                    frame_count += 1

                    print(f'Time Stamp {time_stamp_count}: {time_stamp[0]} --> {time_stamp[1]}')
                break
            else:
                print(f'{character} is not a valid character. Please chose another character.')

        print(f'{character} was found {frame_count} times in the script')
        print()

        char = character_info.lower()
        directory = f'frame_images/{character_info}'
        image_dataset(directory, character_info, film)
        break

    elif char_or_object.upper() == 'O':
        print('Objects found from our text analysis:')
        for obj in object_names:
            print(f'{obj}')
        print()

        while True:
            obj = input("Enter the object you would like to display: ")
            print()
            object_info = obj.upper()

            frame_count = 0
            time_stamp_count = 0

            if object_info in object_dict:
                print(f'Time stamps for {obj}:')
                for time_stamp in object_dict[object_info]:
                    time_stamp_count += 1
                    frame_count += 1

                    print(f'Time Stamp {time_stamp_count}: {time_stamp[0]} --> {time_stamp[1]}')
                break
            else:
                print(f'{obj} is not a valid object.  Please enter a valid object')

        print(f'{obj} was found {frame_count} times in the script.')
        print()

        obj = object_info.lower()
        directory = f'frame_images/{object_info}'
        image_dataset(directory, object_info, film)
        break

    else:
        print('Not a valid option. Please chose either "C" or "O"')
