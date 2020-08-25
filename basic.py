import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.base import clone

import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
import spacy

import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()


np.set_printoptions(precision = 2)


def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def make_dir(outp):
    if not os.path.exists(outp):
        print("[Creating output directory in "+outp+"]")
        os.makedirs(outp)

CORPUS = '../AMI2020_TrainingSet/AMI2020_training_raw.tsv'
DATA_FRAME = pd.read_csv(CORPUS, '\t')

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


def pos_ngrams(documents, N=3, M=3, lang='it'):
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


model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, max_iter=10000, 
                         C=3, fit_intercept=True, intercept_scaling=1.0, 
                         solver = 'liblinear', warm_start=False,
                         class_weight=None, random_state=None)


vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
    ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1)


data = DATA_FRAME['text']
X = vectorizer.fit_transform(data)


acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = cv_with_combined(model, 10, X, DATA_FRAME['misogynous'], DATA_FRAME['aggressiveness'], runs=10)
print("combined")
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)




print("basic tfidf pe misogy ")
labels = DATA_FRAME['misogynous']
X = vectorizer.fit_transform(data)
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)



print("basic tfidf pe aggressive")
X = vectorizer.fit_transform(data)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)



#################
cvc = CountVectorizer(min_df=3, max_features=None, 
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 2))#, vocabulary=fun_fr)
X = cvc.fit_transform(data)
print ("doing log_entropy")
X_log = log_entropy(X)

print("log entropy all words")
labels = DATA_FRAME['misogynous']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)




print ("doing pos_ngrams")
X, X_tf, _ = pos_ngrams(data, N=2, M=3, lang='it')
print("pos ngrams")
labels = DATA_FRAME['misogynous']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)


print("tf idf on pos ngrams")
labels = DATA_FRAME['misogynous']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X_tf, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X_tf, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)


X = log_entropy(X)
print("log entropy on pos ngrams")
labels = DATA_FRAME['misogynous']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=10)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)

'''
''' 