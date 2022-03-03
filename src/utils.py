#Import the libraries
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle
import string
import json
from gensim.parsing.preprocessing import remove_stopwords
from wordcloud import WordCloud,STOPWORDS
import streamlit as st
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Import Word2Vec models for Continous bag of words and Skipgram created from Jupyter notebook
model_cbow =Word2Vec.load('../data/w2v_fp.model')

model_sg =Word2Vec.load('../data/w2v_sg.model')

# Import the word list of possible words related to Bhagavad Gita 
with open('../data/list_gita.txt') as d:
    list_words=d.read().split('\n')

# Importing different dataframes to be used by the app
df2 =pd.read_csv('data/df_bhagwad_gita.csv')
df2.fillna('', inplace=True)
# print(df2.head())
df_verse_list= pd.read_csv('data/verse_list.csv')
# print(df_verse_list.head())
df_bg_kaggle=pd.read_csv('data/bhagavad-gita.csv')
df_bg_kaggle.rename(columns={'title':'no. of verses'},inplace=True)
# print(df_bg_kaggle.head())
df_chp_desc=pd.read_csv('data/df_chp_desc.csv')

def chapter_desc(chapter):
    chapter=int(chapter)
    text = df_chp_desc['description'][chapter-1]
    return text

def wordcloud_eng(chapter):
    chapter =int(chapter)
    if chapter ==1:
        start = 0
    else:
        start =df_verse_list['verses_count'][chapter-2]
    end = df_verse_list['verses_count'][chapter-1]
    chap_df=df2[start:end]
    # create the wordcloud
    words = chap_df['commentary']
    mask = np.zeros((500, 500, 3), np.uint8)
    mask[150:350,150:350,:] = 0
    wordcloud1 =WordCloud(max_font_size=75,
                         width =600,
                         height=600,
                         background_color="white",
                         max_words =100).generate(' '.join(words))
    
    return wordcloud1
    
def wordcloud_sans(chapter):
    chapter =int(chapter)
    if chapter ==1:
        start = 0
    else:
        start =df_verse_list['verses_count'][chapter-2]
    end = df_verse_list['verses_count'][chapter-1]
    chap_df=df2[start:end]
    # create the wordcloud
    words = chap_df['transliteration']
    mask = np.zeros((500, 500, 3), np.uint8)
    mask[150:350,150:350,:] = 0
    wordcloud2 =WordCloud(max_font_size=75,
                         width =600,
                         height=600,
                         background_color="white",
                         max_words =100).generate(' '.join(words))
    
    return wordcloud2

def model_cmp(mdl,text):
    if mdl=="CBOW":
        res =model_cbow.wv.most_similar(text.split(),topn=20)
        df_res =pd.DataFrame(res)
        df_res.rename(columns={0:'similar words',1:'cosine similarity'},inplace=True)
        
    elif mdl=="skipgram":
        res = model_sg.wv.most_similar(text.split(),topn=20)
        df_res =pd.DataFrame(res)
        df_res.rename(columns={0:'similar words',1:'cosine similarity'},inplace=True)
    else: 
        res =[]
    return df_res

Chapter_Topics =[]
for i in range (0,18):
    chp_num = (i+1)
    chp_name = df_chp_desc['description'][i]
    chp_txt = str(chp_num) + ": " + chp_name
    Chapter_Topics.append(chp_txt)