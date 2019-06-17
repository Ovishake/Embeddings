import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import matplotlib
from matplotlib import pyplot as plt

import gensim
import nltk
import string

from nltk.corpus import stopwords

import re

stop_words = set(nltk.corpus.stopwords.words('english'))

from nltk import word_tokenize, sent_tokenize

from gensim.models import Word2Vec

from nltk.corpus import gutenberg

from sklearn.manifold import TSNE

import plotly.plotly as py
import plotly.graph_objs as go

sample_text = gutenberg.raw('austen-emma.txt')
print(type(sample_text))

sample_text = sample_text.encode("utf-8")

sentences = nltk.sent_tokenize(sample_text)

print("Number of sentences", len(sentences))

token_sent = []

#translation_rule = str.maketrans('','', string.punctuation)


for sent in sentences:
    #This is how you strip text of punctuation in Python 3.x
    sent = re.sub('['+string.punctuation+']', '', sent)
    words = nltk.word_tokenize(sent)
    words = [w for w in words if w not in stop_words]
    token_sent.append(words)
    
print(len(token_sent))

w2v_model = Word2Vec(token_sent, size=100, min_count=1, window=10, sg = 1, hs = 0, seed=42, workers = 4)
w2v_model.train(token_sent, total_examples= len(token_sent), epochs=10)

vocab = list(w2v_model.wv.vocab)

print(type(vocab))

print(vocab)

print(w2v_model.wv.get_vector('looking'))

looking = w2v_model.wv.get_vector('looking')

w2v_model.wv.most_similar('Miss', topn=15)

#testing the similarity
w2v_model.wv.similarity('beloved', 'friend')

w2v_model.wv.similarity('equal', 'equal')

w2v_model.save('word2vec.pickle')

model = Word2Vec.load('word2vec.pickle')

print(model)

print(w2v_model)

embed_matrix = []

for word in vocab:
    embed_matrix.append(w2v_model.wv.get_vector(word))
    
tsne = TSNE(n_components=2, random_state=42)
T = tsne.fit_transform(embed_matrix)

#PLotly public api-key: RV1Ct3uJgjT1kxPeZgLu
#Account Name: Ovishake

py.sign_in('Ovishake', 'RV1Ct3uJgjT1kxPeZgLu')

#create a trace

trace = go.Scatter(
        x = T[:,0],
        y = T[:,1],
        mode='markers',
        text=list(w2v_model.wv.vocab))

data = [trace]

py.iplot(data, filename= 'PlotlyScatterPlot')


    

