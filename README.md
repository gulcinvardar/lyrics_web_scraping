# lyrics_web_scraping

A web scraping program to get lyrics from any artist archived in 'https://lyrics.az/'

### Usage

- get_lyrics_from_any_artist.py

    The lyrics of the songs of the artist is saved in a dataframe for further usage. Saves it as csv.
    While crawling in the web-site, it prints which song link is being scraped.
    

- song_lyrics_with_artist_of_choice_model.py

    Select two artists from the saved csv files. Give the file names as "artist1.csv"
    The program cleans the lyrics, lemmatizes and lemmatizes them.
    Logistic regression is used to predict one of the two selected artists using the new lyrics entered by the user.
    It pickles the vectorizer and the model to be used later in CLI as an artist prediction game.

- predict_amy_or_ratm_pickle.py
    An example how to use the pickled vectorizer and the model to predict the artits using CLI.


### License

(c) 2022 Gülçin Vardar

Distributed under the conditions of the MIT License. See LICENSE for details.