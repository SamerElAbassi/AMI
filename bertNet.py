import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from nltk.corpus import stopwords
import spacy


def clean_text(text):
    # Convert words to lower case
    text = text.lower()

    # Format words and remove unwanted characters

    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Remove stopwords
    text = text.split()
    stops = set(stopwords.words('italian'))
    text = [w for w in text if not w in stops or 'non']
    text = [w if w not in training_identity else training_identity[0] for w in text]
    text = " ".join(text)

    # Tokenize each word
    text = nltk.WordPunctTokenizer().tokenize(text)
    return text


nlp = spacy.load("it")

training_identity = []
with open('AMI2020_TrainingSet/AMI2020_training_identityterms.txt') as f:
    for word in f.read().split():
        training_identity.append(word)

raw = pd.read_csv('AMI2020_TrainingSet/AMI2020_training_raw.tsv', sep='\t')
keyword_lists = list(map(clean_text, raw.text))
raw['clean'] = [' '.join(l) for l in keyword_lists]

for index, line in enumerate(raw['clean']):
    lem = nlp(line)
    raw['clean'][index] = ' '.join([word.lemma_ for word in lem])

print(raw)

raw_misogynous = raw[raw['misogynous'] == 1]
raw_not_misogynous = raw[raw['misogynous'] == 0]
del raw_misogynous['aggressiveness']
del raw_misogynous['text']
del raw_not_misogynous['aggressiveness']
del raw_not_misogynous['text']

train_test_ratio = 0.10
train_valid_ratio = 0.80

first_n_words = 200

df_real_full_train, df_real_test = train_test_split(raw_misogynous, train_size=train_test_ratio, random_state=1)
df_fake_full_train, df_fake_test = train_test_split(raw_not_misogynous, train_size=train_test_ratio, random_state=1)

# Train-valid split
df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size=train_valid_ratio, random_state=1)
df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size=train_valid_ratio, random_state=1)

# Concatenate splits of different labels
df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)

# Write preprocessed data
df_train.to_csv('./train.csv', index=False)
df_valid.to_csv('./valid.csv', index=False)
df_test.to_csv('./test.csv', index=False)

# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('misogynous', label_field), ('clean', text_field)]  # Maybe try with text@

train, valid, test = TabularDataset.splits(path='.data/', train='train.csv', validation='valid.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.clean),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.clean),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=False, sort=False)


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "dbmdz/bert-base-italian-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, clean, label):
        loss, clean_fea = self.encoder(clean, labels=label)[:2]

        return loss, clean_fea


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train(model,
          optimizer,
          criterion=nn.BCELoss(),
          train_loader=train_iter,
          valid_loader=valid_iter,
          num_epochs=5,
          eval_every=len(train_iter) // 2,
          file_path='.data/',
          best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (labels, clean),_ in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            clean = clean.type(torch.LongTensor)
            clean = clean.to(device)
            output = model(clean, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (labels, clean), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device)
                        clean = clean.type(torch.LongTensor)
                        clean = clean.to(device)
                        output = model(clean, labels)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)
