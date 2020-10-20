import nltk
import os
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

import pandas as pd
from scipy.sparse import hstack
import spacy


from it_fun_wds import FUN_WDS
from noun_chunks import *
from check_glove_italian import read_glove


# more effective to preload the language first
lang='it_core_news_lg'
nlp = spacy.load(lang)

GLV = read_glove('glove/glove.twitter.27B.200d_small.txt')
#GLV = read_glove('glove/glove.twitter.27B.200d.txt')

np.set_printoptions(precision = 2)


def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def make_dir(outp):
    if not os.path.exists(outp):
        print("[Creating output directory in "+outp+"]")
        os.makedirs(outp)


def average_confusion(confusion_matrices):
    avg = np.mean(confusion_matrices, axis=0)
    total = sum(avg)
    return avg*100/total

def run_cv(classifier, k_fold, data, labels, runs=10):
    accuracy_scores = []
    f1scores = []
    auc_scores = [0]
    confusion_matrices = []
    min_acc = 100
    labels = np.array(labels)
    skf = StratifiedKFold(k_fold)
    for run in range(0, runs):
        cv_splits = skf.split(data, labels)
        for train, test in cv_splits:
            traindata = data[train]
            y_traindata = labels[train]
            testdata = data[test]
            y_testdata = labels[test]
            model = clone(classifier)
            model.fit(traindata, y_traindata)
            result = model.predict(testdata)
            score = accuracy_score(y_testdata, result)
            accuracy_scores.append(score)
            f1sc = f1_score(y_testdata, result, average='weighted')  
            f1scores.append(f1sc)
            #auc = roc_auc_score(y_testdata, result, average='weighted', multi_class='ovr')
            #auc_scores.append(auc)
            if f1sc < min_acc:
                min_acc = f1sc 
                split_inidices = (train, test)
            confusion_matrices.append(np.array(confusion_matrix(y_testdata, result)))
    print ("min cv F1: ", min_acc)   
    return (accuracy_scores, f1scores, auc_scores, average_confusion(confusion_matrices), split_inidices)

def add_to_features(X, feature_values):
    extra_ftrs = np.matrix(feature_values)
    return hstack((X, extra_ftrs.T)).tocsr()

def build_aggressivensess_calssifier(classifier, data, aggr):
    model = clone(classifier)
    model.fit(data, aggr)
    return model    

def cv_with_combined(classifier, k_fold, data, misog, agresiv, runs=10):
    accuracy_scores = []
    f1scores = []
    auc_scores = [0]
    confusion_matrices = []
    min_acc = 100
    misog = np.array(misog)
    agresiv = np.array(agresiv)
    skf = StratifiedKFold(k_fold)
    for run in range(0, runs):
        cv_splits = skf.split(data, misog)
        for train, test in cv_splits:
            traindata = data[train]
            aggr_classifier = build_aggressivensess_calssifier(classifier, traindata, agresiv[train])
            traindata = add_to_features(traindata, agresiv[train])
            y_traindata = misog[train]
            testdata = data[test]
            agresiv_pred = aggr_classifier.predict(testdata)
            testdata = add_to_features(testdata, agresiv_pred)
            y_testdata = misog[test]
            model = clone(classifier)
            model.fit(traindata, y_traindata)
            result = model.predict(testdata)
            score = accuracy_score(y_testdata, result)
            accuracy_scores.append(score)
            f1sc = f1_score(y_testdata, result, average='weighted')  
            f1scores.append(f1sc)
            #auc = roc_auc_score(y_testdata, result, average='weighted', multi_class='ovr')
            #auc_scores.append(auc)
            if f1sc < min_acc:
                min_acc = f1sc 
                split_inidices = (train, test)
            confusion_matrices.append(np.array(confusion_matrix(y_testdata, result)))
    print ("min cv F1: ", min_acc)   
    return (accuracy_scores, f1scores, auc_scores, average_confusion(confusion_matrices), split_inidices)


def get_noun_phrases_embeddings(documents):
    phrased = []
    for idx, text in enumerate(documents):
        doc = nlp(text)
        all_vectors = [doc[start:end].vector for start,end,_ in noun_chunks(doc)]
        if not all_vectors:
            all_vectors = [np.zeros((300,))]
        phrases = np.mean(all_vectors, axis=0)
        phrased.append(phrases)
    return np.array(phrased)

def get_spacy_embeddings(documents):
    phrased = []
    for idx, text in enumerate(documents):
        doc = nlp(text)
        phrased.append(doc.vector)
    return np.array(phrased)


def get_mean_embeddings(texts, embeddings=GLV):
    means = []
    dim = len(list(embeddings.values())[0])
    for text in texts:
        embs = [np.zeros(dim)]
        text = nltk.WordPunctTokenizer().tokenize(text)
        embs.extend([embeddings[w] if w in embeddings else np.zeros(dim) for w in text])
        mn = np.mean(embs, axis=0)
        means.append(mn)
    return np.array(means)



def get_ftrs(tag):
    ftrs = tag.split('|')
    vals = []
    for ft in ftrs:
        parts = ft.split('=')
        if len(parts) > 1:
            vals.append(parts[1])
        else:
            vals.append(ft) 
    return ' '.join(vals)

def pos_ngrams(documents, N=2, M=3):
    pos_docs = []
    for idx,text in enumerate(documents):
        #print("Extracting POS for ", idx)
        doc = nlp(text)
        #pos_tags = " ".join([t.tag_ for t in doc])
        pos_tags = " ".join([get_ftrs(t.tag_) for t in doc])
        pos_docs.append(pos_tags)
    c = CountVectorizer(ngram_range=(N,M))
    tfidf = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(N, M), use_idf=1, smooth_idf=1, sublinear_tf=1)
    features = c.fit_transform(pos_docs)
    tfidf_ftrs = tfidf.fit_transform(pos_docs)
    return features, tfidf_ftrs, c

def pos_ngrams_filtered(documents, filtered_pos, N=1, M=5):
    pos_docs = []
    for idx,text in enumerate(documents):
        #print("Extracting POS for ", idx) 
        doc = nlp(text)
        ftrs = []
        ftrs = [get_ftrs(t.tag_) for t in doc if t.pos_ in filtered_pos]
        ftrs.extend([t.text for t in doc if t.pos_ in filtered_pos])
        pos_tags = " ".join(ftrs)
        pos_docs.append(pos_tags)
    c = CountVectorizer(ngram_range=(N,M))
    tfidf = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(N, M), use_idf=1, smooth_idf=1, sublinear_tf=1)
    features = c.fit_transform(pos_docs)
    tfidf_ftrs = tfidf.fit_transform(pos_docs)
    return features, tfidf_ftrs, c

def pronouns(documents, N=1, M=5):
    filtered_pos = {'PRON'}
    return pos_ngrams_filtered(documents, N, M, filtered_pos=filtered_pos)

def log_entropy(matrix):
    if type(matrix) is not np.ndarray:
        matrix = matrix.toarray()
    normalized = matrix / (1 + np.sum(matrix, axis=0))
    nr_docs, _ = matrix.shape
    '''
        g_i = 1 + sum     p_ij * log(p_ij + 1)   
                 j=1,N  ------------------------
                               log(N)                              
    '''
    entropy = 1 + np.sum(np.multiply(normalized, np.log(normalized + 1)), axis=0)/np.log(nr_docs)
    '''
        logent_ij = gi * log(tf_ij + 1)
    '''
    log_ent = entropy * np.log(matrix + 1)
    return log_ent




def clean_text(text):
    text = text.lower()
    text = text.split()
    text = [w for w in text if '@' not in w]
    text = [w for w in text if 'http' not in w]
    return " ".join(text)

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, max_iter=100000,
                         C=3, fit_intercept=True, intercept_scaling=1.0, 
                         solver = 'liblinear', warm_start=False, 
                         class_weight=None, random_state=None)


#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
#    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
#    ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1)


vectorizer = TfidfVectorizer(min_df=3,  max_features=None,
    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
    ngram_range=(1, 5), use_idf=True)#, vocabulary = FUN_WDS)




train_df = pd.read_csv('AMI2020_TrainingSet/AMI2020_training_raw_nps.tsv', '\t')
test_df = pd.read_csv('AMI2020_TestSet/AMI2020_test_raw_gt_nps.tsv', '\t')

key_col = 'text'
#key_col = 'nps'

#clean
train_df[key_col] = train_df[key_col].apply(clean_text)
test_df[key_col] = test_df[key_col].apply(clean_text)

train_data = train_df[key_col].values

y_train_misog = train_df['misogynous']
y_train_aggrs = train_df['aggressiveness']


test_data = test_df[key_col].values

y_test_misog = test_df['misogynous']
y_test_aggrs = test_df['aggressiveness']

data = list(train_data) + list(test_data)

#data = get_noun_phrases_for_docs(data)
#X_cnt, X_tf, _ = pos_ngrams(data, N=2, M=7)
#mix_pos = {'NOUN', 'ADV', 'ADP', 'DET', 'ADJ', 'VERB', 'AUX', 'PRON'}
#X_cnt, X_tf, _ = pos_ngrams_filtered(data, filtered_pos=mix_pos)
#X_cnt, X_tf, _ = pronouns(data)
#X = X_tf[:len(train_data)]
#X_test = X_tf[len(train_data):]

vectorizer.fit(data)
X = vectorizer.transform(data[:len(train_data)])
X_test = vectorizer.transform(data[len(train_data):])

#nps_embds = get_noun_phrases_embeddings(data)
#nps_embds = get_spacy_embeddings(data)
#nps_embds = get_mean_embeddings(data)
#print(nps_embds.shape)
#X = nps_embds[:len(train_data)]
#X_test = nps_embds[len(train_data):]

#default_grid = {"C": [0.001, 0.01, 0.1, 1, 2, 10]}
#model = GridSearchCV(model, default_grid, cv= StratifiedKFold(5), n_jobs=3)

model.fit(X, y_train_misog)
misog_preds = model.predict(X_test)

model.fit(X, y_train_aggrs)
aggrs_preds = model.predict(X_test)
for idx, (mpre, apre) in enumerate(zip(misog_preds, aggrs_preds)):
    if mpre==0 and apre==1:
        aggrs_preds[idx] = 0

_, f1_scores_misog, _, _, _ = run_cv(model, 10, X, y_train_misog, runs=1)
_, f1_scores_aggre, _, _, _ = run_cv(model, 10, X, y_train_aggrs, runs=1)
f1sc_m = f1_score(y_test_misog, misog_preds, average='macro')
f1sc_a = f1_score(y_test_aggrs, aggrs_preds, average='macro')  
macro_score = (f1sc_m + f1sc_a) / 2

print ("\tAverage F1 score misog", np.mean(f1_scores_misog), ' ', np.std(f1_scores_misog))
print ("\tF1 score misog", f1sc_m)
print ("\tAverage F1 score aggressiveness", np.mean(f1_scores_aggre), ' ', np.std(f1_scores_aggre))
print ("\tF1 score aggressiveness", f1sc_a)
print ("\tF1 Macro score", macro_score)

cols = ['cv msgny', 'f1 msgny', 'cv agrsv', 'f1 agrsv', 'f1 macro']
vals = ["{:.3f}".format(np.mean(f1_scores_misog)),
"{:.3f}".format(np.mean(f1sc_m)),
"{:.3f}".format(np.mean(f1_scores_aggre)),
"{:.3f}".format(np.mean(f1sc_a)),
"{:.3f}".format(macro_score)]
print('\t'.join(cols))
print('\t'.join(vals))

'''
values = []
for idx, mpre, apre in zip(range(5001, 6001), misog_preds, aggrs_preds):
    row = {}
    row['id'] = idx
    row['misogynous'] = mpre
    row['aggressiveness'] = apre
    values.append(row)
out_df = pd.DataFrame(values)
out_df.to_csv('MDD.run1', index=False, sep='\t', header=False)
'''