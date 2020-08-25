'''
exemplu de extractie noun chunks
'''
import spacy
nlp = spacy.load('it')
txt = "Puttana — ma che cazzo ne sai tu ma la smetti"
doc = nlp(txt)
from noun_chunks import *
ncs = noun_chunks(doc)
chunks = [doc[nc[0]:nc[1]].text.lower() for nc in ncs]
text_chunks = " ".join(chunks)
#TODO:
# 1 let's try tf-idf classification using text as noun chunks
# 2 let's try text classification with noun chunks as they are