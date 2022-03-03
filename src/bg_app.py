import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from utils import df_bg_kaggle,df_verse_list,df2,Chapter_Topics
from word_network import network_graph
from wordcloud import WordCloud,STOPWORDS
from utils import wordcloud_eng,wordcloud_sans,chapter_desc,model_cmp
import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx
# showWarningOnDirectExecution = True
# Add title and Introduction
st.set_option('deprecation.showPyplotGlobalUse', False)

nav_rad = st.sidebar.radio("Contents", ["Home","Wordclouds","word2vec models","Interactive word embedding"])
if nav_rad== "Home":

    st.title(" WordClouds and word embedding visualization  on Bhagavad Gita")

    st.subheader(" A Demo app to visualize the contents of the Bhagavad Gita using word2vec and network graphs")
    st.markdown( """ - Data is collected based on the explaination, translation and transliteration of 700 verses of Bhagavad Gita.
- The word clouds are created from the explaination and transliteration words extracted for each chapter
- Semantic /contextual word modeling by:
    - creating the word2vec model out of the word corpus using the gensim library. (522000 words)
    - modeling for (Networkx) nodes and edges of a list of words from the word vectors which the model creates.
    - Visualizing this information through interactive plotly plot.
""")


    st.subheader(" An overview of verses and chapters in the Bhagavad Gita ")
    st.markdown( """ - The Bhagavad Gita is a part of great indian epic: Mahabharata.
- It contains 700 verses (Shlokas) in 18 chapters out of the almost 100000 verses in Mahabharatha.
- The Bhagavad Gita is based on dialogue between Pandava prince and warrior Arjun and his charioteer friend and lord Krishna
- The topics discussed include duties, devotion, righteousness, renunciation amongst many others.
- The Bhagavad Gita is a sacred and significant contribution of Mahabharatha.
""")

    chp =df_bg_kaggle['no. of verses'].astype(int)
    fig = px.histogram(chp,title=" Verses per chapter in Bhagavad Gita",labels ={'value':'chapters','count':'verses/chapter'})
    fig.update_layout(bargap=0.2)
    st.image('data/bg_pic.jpg')
    st.markdown("""###### image credits:https://wallpaper-house.com/wallpaper-id-99084.php ######""")
    st.plotly_chart(fig)
    st.subheader(" Overview of topics covered in the chapters")
    for i in range(0,18):
        st.markdown(Chapter_Topics[i])

if nav_rad== "Wordclouds":


    st.title(" Explore the contents of the chapter through word clouds")
    st.write("Pick a chapter amongst the 18 chapters to explore the contents")
    st.write("(the two word clouds are just a representation of collection of words from English translation and English transliteration (w/o dialect) from the chapter)")


    chapter=st.slider(" Scroll to select the chapter to look into ",1,18)

    st.header("Chapter: " + str(chapter))
    st.subheader(chapter_desc(chapter))

    col1,col2 =st.columns(2)

    with col1:
        st.write(" English ")
            #plot1
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud_eng(chapter),interpolation='bilinear')
        plt.title("Chapter " + str(chapter))
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        st.pyplot()

    with col2:
        st.write("Transliteration")
        #plot2
        plt.figure(figsize=(6,6))
        plt.imshow(wordcloud_sans(chapter),interpolation='bilinear')
        plt.title("Chapter " + str(chapter))
        plt.tight_layout()
        plt.axis("off")
        plt.show()
        st.pyplot()


if nav_rad== "word2vec models":
    st.title(" Understanding the difference between CBOW and skipgram models")
    st.markdown(" **CBOW-> continous bag of words** and skip gram have slightly different approach towards vectorizing a word")
    st.markdown ( "**CBOW** creates context of a word based on the surrounding words while **skipgram** creates context of the surrounding words based on the word ")
    st.image('data/word2vec.jpeg')
    col3,col4=st.columns(2)
    with col3:
        st.write("English words taken from the Bhagavad Gita")
        st.image('data/all_chapters.png')
    
    with col4:
        st.write(" transliteration of Sanskrit taken from the Gita")
        st.image("data/all_chapters_sans.png")
    

    text=st.text_input("Enter some search words from Bhagavad Gita ( like the example below, only small letters, no commas, only space between the words! Thanks :-)",value= "arjun krishna")
    col1,col2 =st.columns(2)
    with col1:
        st.subheader(" CBOW results")
        st.write(" most similar words")
        result = model_cmp("CBOW",text)
        st.dataframe(result)
    
    with col2:
        st.subheader(" skipgram results")
        st.write(" most similar words")
        result = model_cmp("skipgram",text)
        st.dataframe(result)

    st.markdown("""- The Bhagavad Gita is a difficult text to assess the two models because a lot of information is mentioned multiple times.
- The **CBOW** model appears to struggle to understand relationships between different words giving very high score for cosine similarity.
- The **skipgram** model however does slightly better to understand some relationships ,associating certain words closer to each other instead of creating a cluster.
- Check out the interactive word embedding to see the visualization of the clusters created for some words which were tested on the two models.
""")


if nav_rad == "Interactive word embedding":
    st.title(" Word network graphs created from word2vec, Networkx and plotly")  
    st.markdown(""" A list of words ( ~50) are vectorized based on the **CBOW** and **Skipgram** models which were created from words (English and Sanskrit) contained in the Bhagavad Gita.
    The Network Graph of word embeddings shown below contains the words as nodes and connections between them (represented by lines) obtained from the cosine similarity of the vectors of the words.
    The options to choose to create the figure are:
    - CBOW and Skipgram (to vectorize the words) 
    - Filter to limit only words with connections (cosine similarity) higher than a threshold value.
    - Category of words which would be highlighted in the diagram.
    """)

    search = {
            "characters": {'arjun', 'krishna', 'sanjay', 'bheeshma'},
            "places": {'hastinapur','kurukshetre'},
            "thoughts": {'anxiety','devotion','renunciation','happiness','divine'},
            "sanskrit":{'bhava','karma','jnanam','eva'}
    }
    choice =st.radio("Choose CBOW or Skipgram",["CBOW","Skipgram"])
    sims_select=st.slider("Select a filter value for word cosine similarity to be included",0.75,0.95,step =0.05)
    selection=st.selectbox("Select a category",['characters','places','thoughts','sanskrit'])
    searches=search[selection]

    st.plotly_chart(network_graph(searches,sims_select,choice))

