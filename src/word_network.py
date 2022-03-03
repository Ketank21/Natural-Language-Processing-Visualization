#Import the libraries
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from gensim.parsing.preprocessing import remove_stopwords
from wordcloud import WordCloud,STOPWORDS
from gensim.models.word2vec import Word2Vec
from utils import model_cbow,model_sg,list_words
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.offline import plot
import networkx as nx
import random
random.seed(3)

# model_cbow =Word2Vec.load('../data/w2v_fp.model')

# model_sg =Word2Vec.load('../data/w2v_sg.model')

# with open('../data/list_gita.txt') as d:
#     list_words=d.read().split('\n')

def network_graph(searches,sims_select,choice):
    #selecting between trained CBOW and skipgram model
    if choice =="CBOW":
        model = model_cbow
    elif choice =="Skipgram":
        model= model_sg
    else:
        model = model_cbow

    #creating vctors and word list from the words in the word list     
    words, vectors = [], []
    for item in list_words:
        try:
            words.append(item)
            vectors.append(model.wv.get_vector(item))
        except KeyError:
            print(f'Word {item} not found in model!')

    sims = cosine_similarity(vectors, vectors)

    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i<=j:
                sims[i, j] = False




    sims_compare =float(sims_select)

    indices = np.argwhere(sims > sims_compare)


    # Creating network graph with word embedding
    G = nx.Graph()

    for index in indices:
        G.add_edge(words[index[0]], words[index[1]], weight=sims[index[0], index[1]])

    weight_values = nx.get_edge_attributes(G, 'weight')

    positions = nx.spring_layout(G)

    nx.set_node_attributes(G, name='position', values=positions)



    
    # The graph creation 
    ######
    edge_x = []
    edge_y = []
    weights = []
    ave_x, ave_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['position']
        x1, y1 = G.nodes[edge[1]]['position']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        ave_x.append(np.mean([x0, x1]))
        ave_y.append(np.mean([y0, y1]))
        weights.append(f'{edge[0]}, {edge[1]}: {weight_values[(edge[0], edge[1])]}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        opacity=0.7,
        line=dict(width=2, color='White'),
        hoverinfo='text',
        mode='lines')

    edge_trace.text = weights


    node_x = []
    node_y = []
    sizes = []
    for node in G.nodes():
        x, y = G.nodes[node]['position']
        node_x.append(x)
        node_y.append(y)
        if node in searches:
            sizes.append(50)
        else:
            sizes.append(15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            line=dict(color='White'),
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Picnic',
            reversescale=False,
            color=[],
            opacity=0.9,
            size=sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    invisible_similarity_trace = go.Scatter(
        x=ave_x, y=ave_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            opacity=0,
        )
    )

    invisible_similarity_trace.text=weights

    ######
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(adjacencies[0])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    ######
    fig = go.Figure(
        data=[edge_trace, node_trace, invisible_similarity_trace],
        layout=go.Layout(
            title='Network Graph of Word Embeddings',
            template='plotly_dark',
            titlefont_size=20,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            plot_bgcolor='rgb(0,0,0)',
            annotations=[
                dict(
                    text="Adapted from: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) 
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # plot(fig)
    return fig


