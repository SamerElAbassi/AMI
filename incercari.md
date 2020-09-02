# Text processing wise
* Initially I tried to remove the stopwords with the exception of 'non' but that prooved to not be relevant.
* Lemmatization decreased the accuracy.  
* Using noun chuncks decreased the accuracy of the models we tried. Removing the @'s and the links at the end of the tweet also decreased the accuracy. to be investigated why 
 (my theory regarding the link part is that on twitter, the tweets that have a link at the end are the ones with images (the link being the image), maybe misogynists use more pictures? (sexist memes, jokes). The @'s should be split into smaller words, as i have noticed many of them contain offensive language)



# Methods

```
vectorizer=FeatureUnion([
    ('word_vectorizer',TfidfVectorizer(min_df=3,  max_features=20000, 
    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    ngram_range=(2, 3))
),
('char_vectorizer',TfidfVectorizer(max_features=40000, lowercase=True,analyzer='char',ngram_range=(3,6)))
])
```
* TF-IDF gave me a slightly worse percentage than the glove embeddings, but Sergiu's had better results
* Adding char_vectorizer reduced the accuracy


# Models
* Pretty settled on either SVM or Logistic Regression

# Features
* TF-IDF
* Count Vectorizer
* Using the aggressive column for the prediction of misogyny
* Glove Embeddings
* TOTRY:Elmo, a better BERT, combinations
### Interesting Reads

https://github.com/ytnvj2/DocumentEmbedding/blob/master/TFIDFwithEmbeddings.ipynb 


