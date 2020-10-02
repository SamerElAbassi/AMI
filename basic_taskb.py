import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.base import clone

import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
import spacy
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()


np.set_printoptions(precision = 2)

default_grid = {"C": [0.001, 0.01, 0.1, 1, 2, 10]}




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
            print("done one")
            traindata = data[train]
            y_traindata = labels[train]
            testdata = data[test]
            y_testdata = labels[test]
            model = clone(classifier)
            sub_skf = StratifiedKFold(3)
            model = GridSearchCV(model, default_grid, cv=sub_skf, n_jobs=3)
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
        print ('run ', run)
        cv_splits = skf.split(data, misog)
        for train, test in cv_splits:
            #print("done one")
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


CORPUS = 'AMI2020_TrainingSet/AMI2020_training_raw.tsv'
#CORPUS = 'confusions_words_misog.csv'
DATA_FRAME = pd.read_csv(CORPUS, '\t',dtype=str)

SYNT = 'AMI2020_TrainingSet/AMI2020_training_synt.tsv'
SYNT_DF = pd.read_csv(SYNT, '\t',dtype=str)

TEST_RAW = pd.read_csv('AMI2020_TestSet/AMI2020_test_raw.tsv', '\t',dtype=str)
TEST_SYNT = pd.read_csv('AMI2020_TestSet/AMI2020_test_synt.tsv', '\t',dtype=str)


model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, max_iter=100000,
                         C=3, fit_intercept=True, intercept_scaling=1.0, 
                         solver = 'liblinear', warm_start=False,
                         class_weight=None, random_state=None)


vectorizer = TfidfVectorizer(min_df=3,  max_features=None, 
    strip_accents='unicode', analyzer='word', token_pattern=r'\b[^\d\W]+\b',
    ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1)


data1=DATA_FRAME['text'].astype(str)
data2=SYNT_DF['text'].astype(str)

data_test_raw = TEST_RAW['text'].astype(str)
data_test_synt = TEST_SYNT['text'].astype(str)

labels1 = DATA_FRAME['misogynous'].values
labels2 = SYNT_DF['misogynous'].values

labels =  np.concatenate([labels1, labels2])
data = np.concatenate([data1, data2])
print(len(data))
print(len(labels))
X = vectorizer.fit(np.concatenate([data, data_test_raw, data_test_synt]))

X = vectorizer.transform(data)
X_test_raw = vectorizer.transform(data_test_raw)
X_test_synt = vectorizer.transform(data_test_synt)


'''
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = cv_with_combined(model, 10, X, DATA_FRAME['misogynous'], DATA_FRAME['aggressiveness'], runs=10)
print("combined")
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
'''

print("basic tfidf pe misogy ")
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=3)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)


model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, max_iter=100000,
                         C=3, fit_intercept=True, intercept_scaling=1.0, 
                         solver = 'liblinear', warm_start=False,
                         class_weight=None, random_state=None)
model = GridSearchCV(model, default_grid, cv= StratifiedKFold(5), n_jobs=3)

model.fit(X, labels)
preds = model.predict(X_test_raw)
#teamName.subtaskname.dataType.runType.runID
values = []
for idx, pred in zip(range(5001, 6001), preds):
    row = {}
    row['id'] = idx
    row['misogynous'] = pred
    values.append(row)
out_df = pd.DataFrame(values)
out_df.to_csv('MDD.B.r.c.run1', index=False, sep='\t')


preds = model.predict(X_test_synt)
#teamName.subtaskname.dataType.runType.runID
values = []
for idx, pred in zip(range(5001, 6001), preds):
    row = {}
    row['id'] = idx
    row['misogynous'] = pred
    values.append(row)
out_df = pd.DataFrame(values)
out_df.to_csv('MDD.B.s.c.run1', index=False, sep='\t')

'''

print("basic tfidf pe aggressive")
X = vectorizer.fit_transform(data)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=3)
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
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=3)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
labels = DATA_FRAME['aggressiveness']
acc_scores, f1_scores, auc_scores, avg_conf, split_inidices = run_cv(model, 10, X, labels, runs=3)
print ("\tAccuracy", np.mean(acc_scores), ' ', np.std(acc_scores))
print ("\tF1 score", np.mean(f1_scores), ' ', np.std(f1_scores))
print ("\tAverage confusion matrix\n", avg_conf)
'''

'''
Interesting thing about POS n grams is that it has some 60% accuracy.
It means then, that mysog texts have speciffic syntactic structures.
TODO: investigate further 


print ("doing pos_ngrams")
X, X_tf, _ = pos_ngrams(data, N=2, M=3, lang='it_core_news_lg')
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