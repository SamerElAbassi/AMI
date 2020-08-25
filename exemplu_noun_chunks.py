'''
exemplu de extractie noun chunks
'''
import spacy
nlp = spacy.load('it')
txt = "Puttana — ma che cazzo ne sai tu ma la smetti"
doc = nlp(txt)
from noun_chunks import *
ncs = noun_chunks(doc)
chunks = [doc[nc[0]:nc[1]] for nc in ncs]
