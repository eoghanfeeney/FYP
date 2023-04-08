# Evaluating the model
import re

import spacy

nlp = spacy.load('trained_character_model')

# read in the text of the script
with open("Normalized_Script.txt", "r") as f:
    text = f.read()

# Characters in the Script
true_characters = ["SHREK", "DONKEY", "CHARMING", "GODMOTHER", "PUSS", "GINGY", "HAROLD", "LILLIAN", "PINOCCHIO",
                   "WOLF", "FIONA", "DORIS", "DRAGON", "MONGO", "JEROME", "ROYAL MESSENGER", "FIONA & DONKEY",
                   "BLIND MOUSE", "SHREK & FIONA", "ANNOUNCER", "MAN", "LITTLE RED RIDING HOOD", "SHREK & HAROLD",
                   "CHEF", "FURNITURE", "MIRROR", "PRICILLA", "CYCLOPS", "MISS FROG", "CEDRIC", "WORKER",
                   "CAPTAIN HOOK", "DOVES", "JILL", "JILL & MAIDENS", "NOBLEMAN", "NOBLEMANS SON", "GUARD",
                   "GODMOTHER & CHARMING", "JOAN", "KNIGHT", "NARRATOR", "LITTLE PIG", "CAPTAIN", "MUFFIN MAN",
                   "GUARDS", "DONKEY & PUSS", "DRONKEYS", "LITTLE MERMAID", "HEADLESS HORSEMAN"]

doc = nlp(text)
character_names = []

# regexp filter
pattern = r'([A-Z &]+):'

# Characters found from the model
for ent in doc.ents:
    if ent.label_ == 'CHARACTER_NAME' and re.match(pattern, ent.text):
        character_name = re.match(pattern, ent.text).group().rstrip(':')
        if character_name not in character_names:
            character_names.append(character_name)

# for ent in doc.ents:
#     if ent.label_ == "CHARACTER_NAME":
#         if ':' in ent.text:
#             char_name = ent.text.rstrip(':')
#         else:
#             char_name = ent.text
#         if char_name not in character_names:
#             character_names.append(char_name)

print(character_names)


# Calculate precision and recall
tp = len(set(character_names) & set(true_characters))
fp = len(set(character_names) - set(true_characters))
fn = len(set(true_characters) - set(character_names))
precision = tp / (tp + fp)
recall = tp / (tp + fn)


print("Precision:", precision)
print("Recall:", recall)





with open("Normalized_Script.txt", "r") as f:
    text = f.read()

nlp = spacy.load('trained_object_model2')

doc = nlp(text)
object_names = []

true_objects = ['STORYBOOK', 'DRAGONS KEEP', 'PRINCESS FIONA TOWER', 'GINGERBREAD HOUSE', 'BASKET', 'DWARVES', 'SHREK SWAMP',
                'WEDDING RING', 'PLANTS', 'MAIL', 'FISH BOWL', 'FANFARE', 'CARRIAGE', 'FAR FAR AWAY SIGN', 'WATERFALL',
                'LIMOUSINE - ESQUE CARRIAGE', 'PALACE', 'LOBSTER', 'ROAST PIG', 'PUPPY', 'BUSINESS CARD',
                'BATTLE - AXE', 'POISON APPLE', 'PIRATES WITCHES', 'FAR FAR AWAY PALACE', 'P MARK TREE', 'HAIRBALL',
                'FACTORY', 'POTION ROOM', 'GIANT CAULDRON', 'HAPPILY POTION', 'ANIMATED CLOCK', 'ANIMATED CANDELABRA',
                'NOBLEMAN POWDERED WIG CLOTHES', 'BOTTLE MILK', 'HANSEL GRETEL', 'TOM THUMB THUMBELINA',
                'SLEEPING BEAUTY', 'FAIRY GODMOTHER PINK CARRIAGE',
                'PINOCCHIO , GINGY , WOLF , LITTLE PIGS , BLIND MICE', 'DIME BAG', 'METAL WAGON', 'TEA CUPS',
                'STONE PRISON', 'DYNAMITE', 'PINK THONG', 'BAKERY', 'FARBUCKS COFFEE', 'FIREBALL', 'WAND']


pattern = r"\[\[(.*?)\]\]"

for ent in doc.ents:
    if ent.label_ == 'OBJECT_NAME' and re.match(pattern, ent.text):
        object_name = re.match(pattern, ent.text).group().rstrip(']]')
        object_name = object_name.replace("[[", "")
        if object_name.upper() not in object_names:
            object_names.append(object_name.upper())

# for ent in doc.ents:
#     if ent.label_ == 'OBJECT_NAME':
#         if '[[' in ent.text:
#             obj_name = ent.text.replace('[[', "").replace("]]", "")
#         else:
#             obj_name = ent.text
#         if obj_name.upper() not in object_names:
#             object_names.append(obj_name.upper())


print(object_names)

# Calculate precision and recall
tp = len(set(object_names) & set(true_objects))
fp = len(set(object_names) - set(true_objects))
fn = len(set(true_objects) - set(object_names))
precision = tp / (tp + fp)
recall = tp / (tp + fn)


print("Precision:", precision)
print("Recall:", recall)