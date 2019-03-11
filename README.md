# Word2Vec implementation on Reuters corpus
## Data Preprocessing
The Reuters dataset has *7k* documents. All these documents are combined together
to make a single document. The words in the document are lower cased. To perform data
preprocessing run the following command:
```
python3 word2vec.py --tokenize True --build_dict True --tokenized_data_file tokenized_data_lemmatized.pkl
```

If you want to add lemmatization and remove the stopwords run the following command:

```
python3 word2vec.py --tokenize True --build_dict True --lemmatize True --tokenized_data_file tokenized_data_lemmatized.pkl --remove_stop_words True
```
The tokenized data will be saved in the file *tokenized_data_lemmatized.pkl* .
You can also do *stemming* by switching *--stemming* True at run time.

## Word2vec Training
To train the *word2vec* model run the following command:
```
python3 word2vec.py --lemmatize True --embedding_dim 50 --train_embeddings True --nb_negative_samples 10 --lr 0.05 --nb_epochs 14
```
Read word2vec.py to learn about more options.

## Pearson cofficient

To calculate $Pearson cofficient$  run the following command:
```
python3 word2vec.py --pearson_cofficient True --word_model 'checkpoints/word_embeddings_10_50dim_with_lemma_v5.1.2.pkl' --context_model 'checkpoints/context_embedding_10_50dim_with_lemma_v5.1.2.pkl'
```

## K-Neighbors
To get the K-nnearest neighbors of the test word, run the following code:
```
python3 word2vec.py --k_NN "Queen" --lemmatize True --k 10
```
