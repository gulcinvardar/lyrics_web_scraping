
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

import nltk
from nltk.corpus import wordnet

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')




def create_df():
    '''Read the files and return the concatenated df
    The artist names will be used to open the file and also to give the prediction and save the pickle
    '''
    df_1 = pd.read_csv('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/exercises/week_4/' + artist1+ '.csv', index_col=0)                                              # read the first artist
    df_1['artist'] = 1                                                                            
    df_2 = pd.read_csv('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/exercises/week_4/' + artist2 + '.csv', index_col=0)                          # read the second artist
    df_2['artist'] = 0                                                                            
    df = pd.concat([df_1, df_2]).drop(columns = ['title', 'links'])                               # create a data frame joining the two artists
    
    return df


def clean_up(df):
    '''Clean the blank, remove digits, regain the censored slang'''
    df = df.drop(df[df['song-lyrics'] == '[Instrumental]'].index)           # remove the word instrumental
    df = df.drop(df[df['song-lyrics'] == 'blank'].index)                    # remove the word blank which comes from empty lyrics sites
    df['song-lyrics'] = df['song-lyrics'].str.replace('\d+', '')            # remove the digits
    df['song-lyrics'] = df['song-lyrics'].str.replace('\w\w\*\*', 'shit')   # get back the sensored swear word
    df['song-lyrics'] = df['song-lyrics'].str.replace('\w\*\*', 'fuck')     # get back the sensored swear word
    df= df.set_index(['artist'])
    
    return df


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,          # these are the labels for the word type that NLTK accepts
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text):
    '''Lemmatize the words based on the word type using get_wordnet_pos'''
    lemmatizer = nltk.stem.WordNetLemmatizer()              # lemmatize the words           
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()       # the words have to be tokenized before lemmatization
    
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)]


def list_to_string(s): 
    """Lemmetizer brings a list, that should be converted back to a string with .join"""
    str1 = " " 
    
    return (str1.join(s))


def lemmatized_str(df):
    '''Lemmatize the words and join them in a string'''
    df['text_lamme'] = df['song-lyrics'].apply(lemmatize)    # create a new column with lemmatized words: brings a list
    df['lyrics'] = df['text_lamme'].apply(list_to_string)      # create a new column with words joined in a string
    df = df.drop(columns= ['song-lyrics', 'text_lamme'])     # drop the unused columns

    return df

def print_evaluations(ytrue, ypred, model_name):
    print(f'Scores of the model {model_name}:')
    print(f'The accuracy of the model is: {round(accuracy_score(ytrue, ypred), 3)}')
    print(f'The precision of the model is: {round(precision_score(ytrue, ypred), 3)}')
    print(f'The recall of the model is: {round(recall_score(ytrue, ypred), 3)}')
    print(f'The f1-score of the model is: {round(f1_score(ytrue, ypred), 3)}')


def vec(X_train, X_test):
    '''Vectorize both the train and the test data'''
    vectorizer = TfidfVectorizer(ngram_range= (1,2), max_df=0.8, min_df= 0.025, strip_accents='ascii', stop_words='english')
    X_matrix = vectorizer.fit_transform(X_train)
    X_train_vec = pd.DataFrame(X_matrix.todense(), columns=vectorizer.get_feature_names())
    X_matrix_test = vectorizer.transform(X_test)
    X_test_vec = pd.DataFrame(X_matrix_test.todense(), columns=vectorizer.get_feature_names())

    return X_train_vec, X_test_vec, vectorizer



def create_model_and_predict(X_train_vec, y_train, X_test_vec):
    ''' Fit Logistic Regression'''
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_vec, y_train)
    ypred = model.predict(X_test_vec)
    print_evaluations(y_test, ypred, 'Logistic')

    return model

def predict_new_lyrics(new_lyrics):
    ''' the lyrics will be given by the user
        they will first be processed and then the artist will be predicted
    '''
    new_lyrics_processed = lemmatized_str(new_lyrics)
    X_new_lyrics = new_lyrics_processed['lyrics']

    new_lyrics_matrix = vectorizer.transform(X_new_lyrics)
    new_lyrics_vec = pd.DataFrame(new_lyrics_matrix.todense(), columns=vectorizer.get_feature_names())
    
    results =  model.predict(new_lyrics_vec)
    probability = model.predict_proba(new_lyrics_vec)

    return probability, results, new_lyrics

def pickle_dump(data, filename):
    ''' load the vectorizer and the model into pickle for later predictions using CLI. Example: file name: predict_amy_or_ratm_pickle.py '''
    pickle.dump(data, open(filename, 'wb'))


# start the program by selecting the artists
artist1 = input('the file name_1: ')
artist2 = input('the file name_2: ')

# create the data frame
df = create_df()
df_clean = clean_up(df)
df_processed = lemmatized_str(df_clean)


# Split the data 
X = df_processed['lyrics']
y = df_processed.index
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

# Fit the model
X_train_vec, X_test_vec, vectorizer = vec(X_train, X_test)
model = create_model_and_predict(X_train_vec, y_train, X_test_vec)


# load the vectorizer and the model into pickle 
pickle_dump(vectorizer, f'{artist1}_{artist2}_vectorizer.pickle')
pickle_dump(model, f'{artist1}_{artist2}_model.pickle')

# play with the artist prediction game and display the results
enter_lyrics = None
while not enter_lyrics == 'Done':
    print(enter_lyrics)
    enter_lyrics = input('Enter lyrics, if you do not want to continue write Done: ')
    new_lyrics_processed = pd.DataFrame([enter_lyrics], columns=['song-lyrics'])
    probability, results, new_lyrics = predict_new_lyrics(new_lyrics_processed)
    if results == 1:
        print (f'{artist1} with a {probability[0][1]} probability')
    else:
        print (f'{artist2} with a {probability[0][0]} probability')




