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


from noun_chunks import *
from evaluation_submission import evaluate_task_b_singlefile

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


def get_noun_phrases_for_docs(documents, lang='it_core_news_lg'):
    nlp = spacy.load(lang)
    phrased = []
    for idx, text in enumerate(documents):
        doc = nlp(text)
        phrases = [doc[start:end].text for start,end,_ in noun_chunks(doc)]
        phrased.append(" ".join(phrases))
    return phrased

def get_noun_phrases_embeddings(documents, lang='it_core_news_lg'):
    nlp = spacy.load(lang)
    phrased = []
    for idx, text in enumerate(documents):
        doc = nlp(text)
        all_vectors = [doc[start:end].vector for start,end,_ in noun_chunks(doc)]
        if not all_vectors:
            all_vectors = [np.zeros((300,))]
        phrases = np.mean(all_vectors, axis=0)
        phrased.append(phrases)
    return np.array(phrased)

def get_spacy_embeddings(documents, lang='it_core_news_lg'):
    nlp = spacy.load(lang)
    phrased = []
    for idx, text in enumerate(documents):
        doc = nlp(text)
        phrased.append(doc.vector)
    return np.array(phrased)


def pos_ngrams(documents, N=3, M=3, lang='it_core_news_lg'):
    nlp = spacy.load(lang)
    pos_docs = []
    for idx,text in enumerate(documents):
        #print("Extracting POS for ", idx)
        doc = nlp(text)
        pos_tags = " ".join([t.pos_ for t in doc])
        pos_docs.append(pos_tags)
    c = CountVectorizer(ngram_range=(N,M))
    tfidf = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(N, M), use_idf=1, smooth_idf=1, sublinear_tf=1)
    features = c.fit_transform(pos_docs)
    tfidf_ftrs = tfidf.fit_transform(pos_docs)
    return features, tfidf_ftrs, c

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


def make_submission_df(preds, tp='raw'):
    if tp == 'raw':
        start = 5001
    else:
        start = 2014
    end = start + len(preds)
    values = []
    for idx, pre in zip(range(start, end), preds):
        row = {}
        row['id'] = idx
        row['misogynous'] = pre
        values.append(row)
    out_df = pd.DataFrame(values)
    return out_df


model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, max_iter=100000,
                         C=3, fit_intercept=True, intercept_scaling=1.0, 
                         solver = 'liblinear', warm_start=False, 
                         class_weight=None, random_state=None)


#vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
#    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
#    ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1)

vectorizer = TfidfVectorizer(min_df=3,  max_features=None,
    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
    ngram_range=(1, 5), use_idf=True)


train_df_raw = pd.read_csv('AMI2020_TrainingSet/AMI2020_training_raw.tsv', '\t')
train_df_synt = pd.read_csv('AMI2020_TrainingSet/AMI2020_training_synt.tsv', '\t')

test_df_raw = pd.read_csv('AMI2020_TestSet/AMI2020_test_raw_gt.tsv', '\t')
test_df_synt = pd.read_csv('AMI2020_TestSet/AMI2020_test_synt_gt.tsv', '\t')

with open('AMI2020_TestSet/AMI2020_test_identityterms.txt', 'r') as fin:
    identity_terms = [term.strip() for term in fin.readlines()]


train_raw = train_df_raw['text'].values
train_synt = train_df_synt['text'].values
y_train_raw = train_df_raw['misogynous']
y_train_synt = train_df_synt['misogynous']


test_raw = test_df_raw['text'].values
test_synt = test_df_synt['text'].values
y_test_raw = test_df_raw['misogynous']
y_test_synt = test_df_synt['misogynous']




data = list(train_raw) + list(test_raw)
vectorizer.fit(data)
X = vectorizer.transform(train_raw)
X_test_raw = vectorizer.transform(test_raw)
X_test_synt = vectorizer.transform(test_synt)

model.fit(X, y_train_raw)
preds_raw = model.predict(X_test_raw)
preds_synt = model.predict(X_test_synt)


prds_synt_df = make_submission_df(preds_synt, 'synt')
prds_raw_df = make_submission_df(preds_raw)

val = evaluate_task_b_singlefile(prds_raw_df, prds_synt_df, test_df_raw, test_df_synt, identity_terms )
print(val)

#data = get_noun_phrases_for_docs(data)
#X_cnt, X_tf, _ = pos_ngrams(data, N=2, M=3, lang='it_core_news_lg')
#X = X_tf[:len(train_data)]
#X_test = X_tf[len(train_data):]

#nps_embds = get_noun_phrases_embeddings(data)
#nps_embds = get_spacy_embeddings(data)
#X = nps_embds[:len(train_data)]
#X_test = nps_embds[len(train_data):]

#default_grid = {"C": [0.001, 0.01, 0.1, 1, 2, 10]}
#model = GridSearchCV(model, default_grid, cv= StratifiedKFold(5), n_jobs=3)


