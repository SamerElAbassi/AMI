# Misogyny Drop Dead
```bash
# download spacy language model for italian and tag it as 'it'
python -m spacy download it_core_news_lg && python -m spacy link it_core_news_lg it
```
### Requirements

All the rest:
```
spacy
scikit-learn
numpy
pandas
```

Bert stuff:
```
torch=1.6
torchtext=0.7
seaborn

```