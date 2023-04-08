import spacy
import tqdm

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")
ner.add_label("OBJECT_NAME")

TRAIN_DATA = [
    ("light shines [[storybook]] . opens , voice turns pages reads . ",
     {"entities": [(13, 26, "OBJECT_NAME")]}),
    ("Charming continues abandoned castle reaches [[Princess Fiona Tower]] .",
     {"entities": [(44, 68, "OBJECT_NAME")]}),
    ("scenes Shrek Fiona honeymoon . break giant [[gingerbread house]] stay night .",
     {"entities": [(43, 64, "OBJECT_NAME")]}),
    ("... arrival [[Shrek swamp]] .",
     {"entities": [(12, 27, "OBJECT_NAME")]}),
    ("Later , Shrek [[dwarves]] forge Fiona [[wedding ring]] . run happily meadow angry mob emerges chasing .",
     {"entities": [(14, 25, "OBJECT_NAME"), (38, 54, "OBJECT_NAME")]}),
    ("SHREK: Oh , mean like … sorting [[mail]] watering [[plants]] ?",
     {"entities": [(32, 40, "OBJECT_NAME"), (50, 60, "OBJECT_NAME")]}),
    ("group uniformed men stand outside , playing [[fanfare]] trumpets drums .",
     {"entities": [(44, 55, "OBJECT_NAME")]}),
    ("silence , cut Fiona throwing luggage [[carriage]]",
     {"entities": [(37, 49, "OBJECT_NAME")]}),
    ("snowy mountains . 200 miles [[Far Far Away sign]] .",
     {"entities": [(28, 49, "OBJECT_NAME")]}),
    ("crossing guard dressed armor brings halt [[limousine esque carriage]] passes .",
     {"entities": [(41, 61, "OBJECT_NAME")]}),
    ("birds released fanfare played Shrek , Fiona , Donkey step [[carriage]] . crowd gasps , fanfare dies",
     {"entities": [(58, 70, "OBJECT_NAME")]}),
    ("Staring grate [[Pinocchio , Gingy , Wolf , Little Pigs , Blind Mice]] .",
     {"entities": [(14, 69, "OBJECT_NAME")]}),
    ("Charming grabs [[wand]] Puss throws Godmother .",
     {"entities": [(15, 23, "OBJECT_NAME")]}),
    ("Donkey throws Gingy Fairy Godmother tries grab . Gingy throws Blind Mice catch . [[wand]] bounces ground activates .",
     {"entities": [(81, 89, "OBJECT_NAME")]}),
    ("Shrek Puss ride Donkey deserted red carpet doors [[palace]] . Suddenly group armed guards block doorway .",
     {"entities": [(49, 59, "OBJECT_NAME")]}),
    ("Shrek points Far Far Away Palace . Mongo instead grabs giant coffee mug [[Farbucks Coffee]] building .",
     {"entities": [(71, 91, "OBJECT_NAME")]}),


]

nlp.begin_training()
for i in tqdm.tqdm(range(20)):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = spacy.training.Example.from_dict(doc, annotations)
        nlp.update([example])

nlp.to_disk("trained_object_model2")


# Testing the model

# doc = nlp("Harold Shrek tug [[roast pig]] middle table , accidentally sending flying upwards .")
# print([(ent.text, ent.label_) for ent in doc.ents])
#
# doc = nlp(" workers caught transformed [[Animated clock]] [[Animated candelabra]] group fleeing elves turned doves")
# print([(ent.text, ent.label_) for ent in doc.ents])

