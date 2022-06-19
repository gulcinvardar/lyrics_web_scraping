
import requests
import re
import time
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

#  Functions:

def get_url(url):
    '''
    Artist name will be given by the user
    If the artist is not found, it will print 'artist not found
    '''
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    html = requests.get(url, headers = headers)
    if html.status_code == 404:
        print('artist not found, try someone else')
        soup = None
    else: 
        soup = BeautifulSoup(html.text, features="html.parser")

    return soup

def get_links_and_titles(soup):
    '''Get the links and the titles into columns in the df'''
    links = []
    titles = []
    lyriclinks = soup.find_all('a',attrs={'class':'py-1'})
    for link in lyriclinks:
        links.append(link['href'])
        titles.append(link['title'].split('-')[1])
    df = pd.DataFrame({'title': titles, 'links':links})
    
    return df


def get_song_lyrics(df):
    '''
    Get the song lyrics with request and BS from the links in the df. 
    While the script is running, it will print the link to see the progress.
    '''
    df['song-lyrics'] = 'blank'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    for i in range(len(df.links)):
        response = requests.get(df.links[i], headers=headers) 
        lyric = BeautifulSoup(response.text, features="html.parser") 
        a = lyric.find('p', attrs={'class': 'song-lyrics'})
        if a is not None:
            df['song-lyrics'][i] = a.text
            print(f'{df.links[i]} crawling in progress')
        time.sleep(3)

    return df


def artist(df):
    '''The artist column will be created with the initial input'''
    df['artist'] = str(singer_corrected)

    return df


def save_df(df):
    '''The file will be saved'''
    df.to_csv(singer_corrected + '.csv')
    print('csv file saved')


# The program starts 

if __name__ == "__main__":
    soup = None 
    while soup is None:
        singer = input("Enter your artist name: ")
        singer_corrected = singer.lower().replace(' ', '-')
        url = 'https://lyrics.az/' + singer_corrected +'/allalbums.html'  
        soup = get_url(url)

    df = get_links_and_titles(soup)
    df = get_song_lyrics(df)
    df = artist(df)
    save_df(df)





