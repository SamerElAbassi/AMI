import os
import pandas as pd
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from flair.data import Corpus
from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from sklearn.model_selection import train_test_split
from torch.optim.adam import Adam


DIR = 'ncs'

#CORPUS = 'AMI2020_TrainingSet/AMI2020_training_raw.tsv'
CORP = 'noun_chunks.csv'
column_name_map = {4: "text", 3: "label"}

CORPUS = os.path.join(DIR, CORP)
DATA_FRAME = pd.read_csv(CORPUS, '\t', dtype=str)
train_dev, test = train_test_split(DATA_FRAME, test_size=0.05, stratify = DATA_FRAME['misogynous'].values)

test.to_csv(os.path.join(DIR, 'test.csv'), index = False, sep = '\t')
train_dev.to_csv(os.path.join(DIR, 'train_dev.csv'), index = False, sep = '\t')

#train_dev.to_csv(os.path.join(DIR, 'train.csv'), index = False, sep = '\t')
#test.to_csv(os.path.join(DIR, 'dev.csv'), index = False, sep = '\t')

train, dev = train_test_split(train_dev, test_size=0.1, stratify = train_dev['misogynous'].values)
train.to_csv(os.path.join(DIR, 'train.csv'), index = False, sep = '\t')
dev.to_csv(os.path.join(DIR, 'dev.csv'), index = False, sep = '\t')

it_embedding = WordEmbeddings('it')


corpus: Corpus = CSVClassificationCorpus(DIR,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter='\t',    # tab-separated files
) 
corpus.filter_empty_sentences()
label_dict = corpus.make_label_dictionary()

word_embeddings = [it_embedding]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=256)

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/taggers/trec',
              learning_rate=0.1,
              mini_batch_size=64,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)