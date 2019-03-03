import numpy as np
import nltk
import re
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from collections import defaultdict

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import argparse
import pickle
from tqdm import tqdm
import math
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('--tokenize', type=bool, default=False)
parser.add_argument('--build_dict', type=bool, default=False)
parser.add_argument('--initilize', type=bool, default=False)
parser.add_argument('--embedding_dim', type=int, default=100)
parser.add_argument('--train_embeddings', type=bool, default=True)
parser.add_argument('--nb_negative_samples', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--context', type=int, default=5)
parser.add_argument('--lemmatize', type=bool, default=True)
parser.add_argument('--stemming', type=bool, default=False)
parser.add_argument('--tokenized_data_file', type=str, default='tokenized_data_lemmatized.pkl')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--lr_annealing', type=bool, default=True)

args = parser.parse_args()
nltk.download('wordnet')
# print(">>> The reuters corpus has {} tags".format(len(reuters.categories())))
# print(">>> The reuters corpus has {} documents".format(len(reuters.fileids())))
lemmer = WordNetLemmatizer()

def sigmoid(x):
    # print(x)
    if x < -500:
        x = -500
    return 1 / (1 + math.exp(-x))

def get_train_test_splits():
    documents = reuters.fileids()
    print(len(documents))
    train_docs = []
    test_docs = []
    for doc in documents:
        split = doc.split('/')[0]
        if split == 'test':
            test_docs.append(doc)
        elif split == 'training':
            train_docs.append(doc)

    print("Test docs: %d Training docs: %d"%(len(test_docs), len(train_docs)))
    return test_docs, train_docs

def get_train_test_docs():
    test_docs, train_docs = get_train_test_splits()
    # print(test_docs)
    test_doc = []
    train_doc = []
    for doc in test_docs:
        test_doc.extend(reuters.words(doc))
    for doc in train_docs:
        train_doc.extend(reuters.words(doc))
    return test_doc, train_doc
nltk.download('stopwords')
cachedStopWords = stopwords.words("english")

def tokenize(lemmatize=False, stemming=False):
    test_doc, train_doc = get_train_test_docs()
    min_length = 3
    test_words = []
    train_words = []
    for word in test_doc:
        test_words.append(word.lower())
    for word in train_doc:
        train_words.append(word.lower())
    print(len(test_words), len(train_words))

    # train_words = map(lambda word: word.lower(), word_tokenize(train_doc))
    # test_words = map(lambda word: word.lower(), word_tokenize(test_doc))


    train_words = [word for word in train_words
                  if word not in cachedStopWords]
    test_words = [word for word in test_words
                  if word not in cachedStopWords]
    if stemming:
        train_tokens =(list(map(lambda token: PorterStemmer().stem(token),
                    train_words)))
        test_tokens =(list(map(lambda token: PorterStemmer().stem(token),
                    test_words)))
    elif lemmatize:
        train_tokens =(list(map(lambda token: lemmer.lemmatize(token),
                    train_words)))
        test_tokens =(list(map(lambda token: lemmer.lemmatize(token),
                    test_words)))

    p = re.compile('[a-zA-Z]+')

    train_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, train_tokens))

    test_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, test_tokens))

    print("Training tokens: %d, test_tokens: %d"%(len(train_tokens), len(test_tokens)))
    data = {'train': train_tokens, 'test': test_tokens}
    if stemming:
        file_name = 'tokenized_data_stemmed.pkl'
    elif lemmatize:
        file_name = 'tokenized_data_lemmatized.pkl'
    else:
        file_name = 'tokenized_data.pkl'
    pickle.dump(data, open(file_name, 'wb'))


def build_dict(file_name):
    print("Reading tokenized train and test documents ++++++++")
    tokenized_data = pickle.load(open(file_name,'rb'))
    train_tokens = tokenized_data['train']
    test_tokens = tokenized_data['test']
    train_counts = Counter(train_tokens)
    test_counts = Counter(test_tokens)
    word2counts = {}
    pickle.dump(train_counts, open('dict_count_train_lemmatized.pkl', 'wb'))
    pickle.dump(test_counts, open('dict_count_test_lemmatized.pkl', 'wb'))

def initilize_word_embeddings(dim):
    print("Initilizing word2vec embeddings ++++++++")
    train_counts = pickle.load(open('dict_count_train_lemmatized.pkl','rb'))
    initial_embeddings = {}

    for word, count in train_counts.items():
        initial_embeddings[word] = np.random.randn(dim)
    return initial_embeddings
    # print("Creating dump of initila word embeddings ++++++++")
    # pickle.dump(initial_embeddings, open('initial_embeddings.pkl', 'wb'))

def normalize_counts(count_dict):
    normalized_counts = {}
    total_count = 0
    for key, item in count_dict.items():
        total_count += item
    for key, item in count_dict.items():
        normalized_counts[key] = item/total_count
    return normalized_counts

def data_generator(data_list, threshold, context, normalized_counts):
    # constant = 601000
    while True:
        constant = 1
        for word_index in range(len(data_list)):
            context_words = []
            if (word_index - (context-1)/2) < 0:
                if np.random.random_sample() > 1 - np.sqrt(threshold/(constant*normalized_counts[data_list[word_index]])):
                    word = data_list[word_index]
                    if len(data_list[0:word_index]) != 0:
                        [context_words.append(c) for c in data_list[0:word_index]]
                    [context_words.append(c) for c in data_list[word_index+1:word_index+int((context+1)/2)]]
                    yield word, context_words
                else:
                    yield None
            elif (word_index + (context-1)/2) >= len(data_list):
                if np.random.random_sample() > 1 - np.sqrt(threshold/(constant*normalized_counts[data_list[word_index]])):
                    word = data_list[word_index]
                    [context_words.append(c) for c in data_list[word_index-int((context-1)/2): word_index]]
                    [context_words.append(c) for c in data_list[word_index+1:]]
                    yield word, context_words
                else:
                    yield None
            else:
                if np.random.random_sample() > 1 - np.sqrt(threshold/(constant*normalized_counts[data_list[word_index]])):
                    word = data_list[word_index]
                    [context_words.append(c) for c in data_list[word_index-int((context-1)/2):word_index]]
                    [context_words.append(c) for c in data_list[word_index+1:word_index+int((context+1)/2)]]
                    yield word, context_words
                else:
                    yield None



def neg_generator(nb_neg_samples, len_dic ,dic_list):
    while True:
        indices = np.random.randint(low=0, high=len_dic, size=nb_neg_samples)
        neg_samples = []
        for index in indices:
            neg_samples.append(dic_list[index])
        yield neg_samples

def get_loss(pos_samples, neg_samples, word_embedding, context_embedding, normalized_counts):
    word = pos_samples[0]
    context = pos_samples[1]

    pos_loss_list = [math.log(sigmoid(np.dot(word_embedding[word], context_embedding[c]))) for c in context]
    # print(pos_loss_list)
    neg_loss_list = [normalized_counts[c]*math.log(sigmoid(-1*np.dot(word_embedding[word], context_embedding[c]))) for c in neg_samples]

    loss = sum(pos_loss_list) + sum(neg_loss_list)
    return loss

def get_grad(pos_samples, neg_samples, word_embedding, context_embedding):
    word = pos_samples[0]
    context = pos_samples[1]

    word_grad = [(1-sigmoid(np.dot(word_embedding[word], context_embedding[c])))*word_embedding[word] for c in context]
    context_grad = defaultdict(list)
    [context_grad[c].append((1-sigmoid(np.dot(word_embedding[word], context_embedding[c])))*context_embedding[c]) for c in context]
    neg_context_grad = defaultdict(list)
    [neg_context_grad[c].append(-1*(1-sigmoid(-1*np.dot(word_embedding[word], context_embedding[c])))*context_embedding[c]) for c in neg_samples]

    return word_grad, context_grad, neg_context_grad, word, context

def sgd_step(embedding, grad, lr):
    pass

def train_word2vec(tokenized_data_file, nb_neg_samples, lr, context, nb_epochs, dim, anneal):
    build_dict(tokenized_data_file)
    train_counts = pickle.load(open('dict_count_train_lemmatized.pkl','rb'))
    normalized_counts = normalize_counts(train_counts)
    threshold = 0.00001
    tokenized_data = pickle.load(open(tokenized_data_file,'rb'))
    word_embedding = initilize_word_embeddings(dim)
    context_embedding = initilize_word_embeddings(dim)
    data_list = tokenized_data['train']
    data_size = len(data_list)
    dic_list = [key for key, item in normalized_counts.items()]
    positive_samples_generator = data_generator(data_list, threshold, context, normalized_counts)
    negative_samples_generator = neg_generator(nb_neg_samples,len(normalized_counts), dic_list)


    for epoch in range(nb_epochs):
        running_loss = []
        print('Epoch {}/{}'.format(epoch, nb_epochs - 1))
        print('-' * 10)
        loader = tqdm(range(data_size), unit='words')
        for i in enumerate(loader):
            positive_samples = next(positive_samples_generator)
            negative_samples = next(negative_samples_generator)
            if positive_samples != None:
                loss = get_loss(positive_samples, negative_samples, word_embedding, context_embedding, normalized_counts)
                word_grad, context_grad, neg_context_grad, word, context = get_grad(positive_samples, negative_samples, word_embedding, context_embedding)
                if anneal:
                    if epoch+1 %5 == 0:
                        lr = 0.1*lr
                for w_grad in word_grad:
                    word_embedding[word] = word_embedding[word] + lr*w_grad
                for key, c_grad in context_grad.items():
                    # print(c_grad[0].shape)
                    context_embedding[key] = context_embedding[key] + lr*c_grad[0]
                for key, n_grad in neg_context_grad.items():
                    context_embedding[key] = context_embedding[key] + c_grad[0]
                running_loss.append(loss)
                loader.set_postfix(loss=sum(running_loss)/len(running_loss))
    pickle.dump(word_embedding, open('word_embeddingsv1.pkl', 'wb'))
    pickle.dump(context_embedding, open('context_embeddingv1.pkl', 'wb'))

                # loss = get_loss(pos_samples, neg_samles)
                # word_grad, context_grad, neg_context_grad, word, context = get_grad(positive_samples, negative_samples, word_embedding, context_embedding)





if args.tokenize:
    tokenize(lemmatize=args.lemmatize, stemming=args.stemming)

if args.build_dict:
    build_dict(args.tokenized_data_file)
if args.initilize:
    initilize_word_embeddings(args.embedding_dim)

if args.train_embeddings:
    train_word2vec(args.tokenized_data_file, args.nb_negative_samples, args.lr,
    args.context, args.nb_epochs, args.embedding_dim, args.lr_annealing)
