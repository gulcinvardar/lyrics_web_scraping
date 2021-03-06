# Web scraping for lyrics and prediction of new lyrics using Logistic Regression

This project was written during Spiced Academy Data Science Bootcamp. 
It is one the weekly projects.

The contents include 
- files to scrape lyrics from any given artist, 
- to create a Logistic Regression model for two selected artists, 
- and an example of prediction of the artist when new lyrics are entered. 


### Usage

- get_lyrics_from_any_artist.py

    It uses 'https://lyrics.az/' to scrape the lyrics of the artist given by the user. 
    The artist name is converted to lower case matching the 'https://lyrics.az/' title style.
    The lyrics are scraped using BeautifulSoup.
    The lyrics of the songs of the artist is inserted into a dataframe and saved as a csv file for further usage.
    While crawling in the web-site, it prints which song link is being scraped.
    

- song_lyrics_with_artist_of_choice_model.py

    Two artists among the saved csv files are used to create a Logistic Regression model. 
    Select two artists from the saved csv files. The file names should be written as the name of the file without the '.csv' extension.
    The first artist is labeled as 1, whereas the second artist is labeled 0. 
    The lyrics in the csv file are read into a dataframe and cleaned up from digits and punctuation. The lyrics were lemmatized using NLTK. 
    The vectorization is done with TfidfVectorizer from SciKit.
    The user is asked to enter new lyrics, that can be also made up. The model predicts which of the two artists is more likely to sing a song with those lyrics. 
    The model prints the predicted artist name with the prediction probability.
    The vectorizer and the model are saved as pickled files for further usage to predict new lyrics. 

- predict_amy_or_ratm_pickle.py

    An example for how to use the pickled vectorizer and the model to predict the artists. 
    In the pickled model, that is created with 'song_lyrics_with_artist_of_choice_model.py', 
    Amy Winehouse was labeled as 1, whereas Rage Against the Machiine was labeled as 0. 
    The program continues as long as the user writes new lyrics as input. 
    To stop the program, write: Done


### License

(c) 2022 G??l??in Vardar

Distributed under the conditions of the MIT License. See LICENSE for details.