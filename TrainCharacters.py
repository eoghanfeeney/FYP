import re
import spacy
import tqdm
from matplotlib import pyplot as plt

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("CHARACTER_NAME")

TRAIN_DATA = [
    ("JILL & MAIDENS: Good morning!", {"entities": [(0, 15, "CHARACTER_NAME")]}),
    ("CHARMING: Oh, thank heavens. Where is she?", {"entities": [(0, 9, "CHARACTER_NAME")]}),
    ("SHREK: Oh, you mean like… sorting the mail and watering the plants?", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("DONKEY: Hey wait a minute, don’t you want to tell me all about your trip? Or how about a game of Parcheesi?",
     {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("MUFFIN MAN: Gingy!", {"entities": [(0, 11, "CHARACTER_NAME")]}),
    ("PINOCCHIO: Shrek? Donkey?", {"entities": [(0, 10, "CHARACTER_NAME")]}),
    ("FIONA: Um, actually--Donkey? Shouldn’t you be getting home ?", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("PUSS: You are told correct, but for this I charge a great deal of money.",
     {"entities": [(0, 5, "CHARACTER_NAME")]}),
    ("MAN: This is it.", {"entities": [(0, 4, "CHARACTER_NAME")]}),
    ("HAROLD: Fairy Godmother. Charming.", {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("SHREK & HAROLD: Here.", {"entities": [(0, 15, "CHARACTER_NAME")]}),
    ("DONKEY & PUSS: Living la vida loca", {"entities": [(0, 14, "CHARACTER_NAME")]}),
    ("GINGY: Not the gumdrop button!", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("GODMOTHER & CHARMING: ''What''?!", {"entities": [(0, 21, "CHARACTER_NAME")]}),
    ("SHREK & FIONA: ''No''!!", {"entities": [(0, 14, "CHARACTER_NAME")]}),
    ("FIONA: [chuckles] Yes!", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("FIONA: No, no, no, Dad! It’s all right! It’s all right. He’s with us. He helped rescue me from the dragon.",
     {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("DRAGON: groans", {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("PUSS: Boss! The Happily Ever After Potion!", {"entities": [(0, 5, "CHARACTER_NAME")]}),
    ("SHREK: Yes.", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("DONKEY: Royal ball?! Can I come?", {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("GINGY: Whizzes on you guys! Hey, mice, pass me a buffalo wing.", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("NOBLEMANS SON: Father? Is everything all right, Father?", {"entities": [(0, 14, "CHARACTER_NAME")]}),
    ("GINGY: It’s a thong!", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("FIONA: Shrek, what are you doing?", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("HAROLD: Well...I guess I gave her the wrong tea.", {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("DRAGON: whimpers", {"entities": [(0, 7, "CHARACTER_NAME")]}),
    ("FIONA: No! They just want to give you their blessing.", {"entities": [(0, 6, "CHARACTER_NAME")]}),
    ("PUSS: The rich King? Sí.", {"entities": [(0, 5, "CHARACTER_NAME")]}),
    ("CYCLOPS: Oh…uh come on in, Your Majesty.", {"entities": [(0, 8, "CHARACTER_NAME")]}),
    ("The Little Mermaid: Ow ow ow ow ahhhhh", {"entities": [(0, 19, "CHARACTER_NAME")]}),
]


nlp.begin_training()
for i in tqdm.tqdm(range(10)):
    for text, annot in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = spacy.training.Example.from_dict(doc, annot)  # trains text and its annots
        nlp.update([example])                                    # adjusts model's weights


nlp.to_disk("trained_character_model")

# Testing the model
# doc = nlp("JILL & MAIDENS: Good morning!")
# print([(ent.text, ent.label_) for ent in doc.ents])

