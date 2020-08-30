import pandas as pd
import spacy
from noun_chunks import *
import nltk

from nltk.corpus import stopwords
nlp = spacy.load('it_core_news_lg')

#Not removing stopwords anymore
def clean_text(text):
    text = text.lower()
    text = text.split()

    text = [w for w in text if '@' not in w]
    text = [w for w in text if 'http' not in w]
    text = " ".join(text)

    # Tokenize each word
    text = nltk.WordPunctTokenizer().tokenize(text)
    return text


CORPUS = 'AMI2020_TrainingSet/AMI2020_training_raw.tsv'
DATA_FRAME = pd.read_csv(CORPUS, '\t')
'''
text_list = []
for i, text in enumerate(DATA_FRAME['text']):
    doc = nlp(text)
    ncs = noun_chunks(doc)
    chunks = [doc[nc[0]:nc[1]].text.lower() for nc in ncs]
    text_chunks = str(" ".join(chunks))
    text_list.append(text_chunks)
DATA_FRAME['clean'] = text_list

#Cleaning up
DATA_FRAME['clean'] = list(map(clean_text, DATA_FRAME['clean']))
DATA_FRAME['clean'] = [' '.join(text) for text in DATA_FRAME['clean']]
DATA_FRAME.to_csv('noun_chuncks,processed.tsv', sep='\t')


del DATA_FRAME['clean']
'''
DATA_FRAME['clean'] = list(map(clean_text, DATA_FRAME['text']))
DATA_FRAME['clean'] = [' '.join(text) for text in DATA_FRAME['clean']]
DATA_FRAME.to_csv('normal_processed.tsv', sep='\t')
