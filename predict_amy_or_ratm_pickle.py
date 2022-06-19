from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    Uses labels for the word type that NLTK accepts.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,          
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text):
    """
    Lemmatize the words based on the word type using get_wordnet_pos
    The words have to be tokenized before lemmatization.
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()                      
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()       # the words have to be tokenized before lemmatization
    
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)]


def list_to_string(s): 
    """Lemmetizer brings a list, that should be converted back to a string with .join"""
    str1 = " " 
    
    return (str1.join(s))


def lemmatized_str(df):
    """Lemmatize the words and join them in a string
    Create a new column with words joined in a string"""
    df['text_lamme'] = df['song-lyrics'].apply(lemmatize)
    df['lyrics'] = df['text_lamme'].apply(list_to_string)
    df = df.drop(columns= ['song-lyrics', 'text_lamme'])

    return df



new_lyrics = pd.DataFrame([input('Enter lyrics: ')], columns=['song-lyrics'])
new_lyrics_processed = lemmatized_str(new_lyrics)['lyrics']

with open("amy_ratm_vectorizer.pickle", "rb") as vector:
    loaded_vector = pickle.load(vector)
lyrics_vector = loaded_vector.transform(new_lyrics_processed)

with open("amy_ratm_model.pickle", "rb") as model:
    loaded_model = pickle.load(model)
artist_prediction = loaded_model.predict(lyrics_vector)
probability = loaded_model.predict_proba(lyrics_vector)

if artist_prediction == 1:
    print (f'{input(" artist1: ")} with a {probability[0][1]} probability')
else:
    print (f'{input(" artist2: ")} with a {probability[0][0]} probability')