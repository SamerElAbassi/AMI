# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# %%
embeddings_index = {}
with open('glove/glove.twitter.27B.200d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs


# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    # Remove stopwords
    text = text.split()
    stops = set(stopwords.words('italian'))
    text = [w for w in text if not w in stops]
    # text = [w if w not in training_identity else training_identity[0] for w in text]
    text = " ".join(text)

    # Tokenize each word
    text = nltk.WordPunctTokenizer().tokenize(text)
    return text


# %%
training_identity = []
with open('AMI2020_TrainingSet\AMI2020_training_identityterms.txt') as f:
    for word in f.read().split():
        training_identity.append(word)

raw = pd.read_csv('different processed dataframes/raw.tsv',sep='\t')

# %%
def get_mean_embeddings(texts, embeddings):
    means = []
    dim = len(list(embeddings.values())[0])
    for text in texts:
        text = nltk.WordPunctTokenizer().tokenize(text)
        means.append(np.mean([embeddings[w] if w in embeddings else np.zeros(dim) for w in text], axis=0))
    return np.array(means)


# %%
def run_cv(k_fold, data, label):
    accuracy_scores = f1_scores = confusion_matrices = []
    labels = np.array(label)
    skf = StratifiedKFold(k_fold)
    cv_splits = skf.split(data, labels)
    min_inidices = ([], [])
    min_acc = 100
    media = 0
    for train, test in cv_splits:
        traindata, y_train, = data[train], labels[train]
        testdata, y_test = data[test], labels[test]
        train_feature_matrix, test_feature_matrix = get_mean_embeddings(data[train], embeddings_index), \
                                                    get_mean_embeddings(data[test], embeddings_index)
        '''
        0.2% mai putin
        model = LogisticRegression(penalty='l2', dual=False, tol=0.0001,
                         C=0.6, fit_intercept=True, intercept_scaling=1.0,
                         solver = 'lbfgs', warm_start=False,
                         class_weight=None, random_state=None)
        '''
        print(train_feature_matrix,train_feature_matrix.shape)
        model = SVC(kernel='rbf', C=1000, gamma=1e-3)
        model.fit(train_feature_matrix, y_train)
        result = model.predict(test_feature_matrix)
        score = accuracy_score(y_test, result)
        accuracy_scores.append(score)
        if score < min_acc:
            min_acc = score
            split_inidices = (train, test)
        f1sc = f1_score(y_test, result, average='weighted')
        media = media + f1sc
        print('f1score:', f1sc)
        f1_scores.append(f1sc)
    print(f'min cv acc:{min_acc}\nmedia:{media}')
    print(np.mean(f1_scores))


# %%
run_cv(10, raw['text'], raw['misogynous'])
