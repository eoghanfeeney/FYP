import spacy

nlp = spacy.load('en_core_web_sm')

# Load the Shrek script
with open('Shrek_2_Script.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Parse text using Spacy nlp object
doc = nlp(text)

# Removing stop words
filtered_tokens = []
for token in doc:
    if not token.is_stop:
        filtered_tokens.append(token)

removed_words = []

for token in doc:
    if token.is_stop and token not in filtered_tokens:
        removed_words.append(token.text)

# Join tokens back together into a string
filtered_text = ' '.join([token.text for token in filtered_tokens])

# Reformatting these desired characters
filtered_text = filtered_text.replace('[ [ ', '[[').replace(' ] ]', ']]')
filtered_text = filtered_text.replace(' : ', ': ')
filtered_text = filtered_text.replace(' > ', '> ')


# Saving normalized script into new file
with open('Normalized_Script.txt', 'w', encoding='utf-8') as file:
    file.write(filtered_text)

print('Removed stop words:')
for word in removed_words:
    print(word)
