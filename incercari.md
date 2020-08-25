Ca text preprocessing initial m am gandit sa nu scot si 'non' din sentences but then i found out it doesnt work like that. lemmatization decreased my accuracy. inca n-am incercat sa scot @-urile ca am zis ca o sa incerc sa le separ candva ca am vazut ca sunt si alea destul de misogine





Ca metode i tried tf-idf vectorizer dar mi-a dat slightly worse decat glove(ceva gen 1%):

```
vectorizer=FeatureUnion([
    ('word_vectorizer',TfidfVectorizer(min_df=3,  max_features=20000, 
    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    ngram_range=(2, 3))
),
('char_vectorizer',TfidfVectorizer(max_features=40000, lowercase=True,analyzer='char',ngram_range=(3,6)))
])
```

Am vaga impresia ca adding char vectorizer imi scazuse sansele.



Din modele i tried random forest, knn, svm si the bert one care e copy pasted de pe net so maybe e bun de ceva if its tweaked but mie mi a dat rau



Deocamdata n am incercat sa pun ca feature si labelurile din agresivitate dar i researched despre classifier chains?? desi nu stiu daca e la fel ca a hard coded solution



I also found this: https://github.com/ytnvj2/DocumentEmbedding/blob/master/TFIDFwithEmbeddings.ipynb and found it interesting si urmeaza sa incerc cateva (in particular 'Combining Word Vectors with TF-IDF to form Sentence Vectors'.



