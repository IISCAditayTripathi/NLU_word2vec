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
from scipy.stats.stats import pearsonr
import operator

nltk.download('reuters')
parser = argparse.ArgumentParser()

parser.add_argument('--tokenize', type=bool, default=False)
parser.add_argument('--build_dict', type=bool, default=False)
parser.add_argument('--initilize', type=bool, default=False)
parser.add_argument('--embedding_dim', type=int, default=50)
parser.add_argument('--train_embeddings', type=bool, default=False)
parser.add_argument('--nb_negative_samples', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--context', type=int, default=7)
parser.add_argument('--lemmatize', type=bool, default=True)
parser.add_argument('--stemming', type=bool, default=False)
parser.add_argument('--tokenized_data_file', type=str, default='tokenized_data_lemmatized.pkl')
parser.add_argument('--nb_epochs', type=int, default=14)
parser.add_argument('--lr_annealing', type=bool, default=False)
parser.add_argument('--simlex_file', type=str, default='/scratche/home/aditay/NLU_assignment1/simlex/SimLex-999/SimLex-999.txt')
parser.add_argument('--pearson_cofficient', type=bool, default=False)
parser.add_argument('--remove_stop_words', type=bool, default=True)
parser.add_argument('--k_NN', type=str, default='king')
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--syntactic_analogy', type=bool, default=False)

args = parser.parse_args()
nltk.download('wordnet')
print("The reuters corpus has {} tags".format(len(reuters.categories())))
print("The reuters corpus has {} documents".format(len(reuters.fileids())))

lemmer = WordNetLemmatizer()

def sigmoid(x):
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

def tokenize(lemmatize=False, stemming=False, remove_stop_words=False):
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

    if remove_stop_words:
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
    else:
        train_tokens = train_words
        test_tokens = test_words

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
        file_name = 'tokenized_data_with_stopwords.pkl'
    pickle.dump(data, open(file_name, 'wb'))


def build_dict(file_name):
    print("Reading tokenized train and test documents ++++++++")
    tokenized_data = pickle.load(open(file_name,'rb'))
    train_tokens = tokenized_data['train']
    test_tokens = tokenized_data['test']
    train_counts = Counter(train_tokens)
    test_counts = Counter(test_tokens)
    word2counts = {}
    print(len(train_tokens))
    pickle.dump(train_counts, open('dict_count_train_lemmatized.pkl', 'wb'))
    pickle.dump(test_counts, open('dict_count_test_lemmatized.pkl', 'wb'))

def initilize_word_embeddings(dim):
    print("Initilizing word2vec embeddings ++++++++")
    train_counts = pickle.load(open('dict_count_train_lemmatized.pkl','rb'))
    initial_embeddings = {}
    sigma = 1
#    sigma = math.sqrt(2/dim)

    for word, count in train_counts.items():
        initial_embeddings[word] = sigma*np.random.uniform(-0.8, 0.8, dim)
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
    return normalized_counts, total_count

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
        # indices = np.random.randint(low=0, high=len_dic, size=nb_neg_samples)
        neg_samples = []
        for i in range(nb_neg_samples):
            index = np.random.randint(low=0, high=len_dic-1)
            neg_samples.append(dic_list[index])
        yield neg_samples

def get_loss(pos_samples, neg_samples, word_embedding, context_embedding, train_counts, total_count):
    word = pos_samples[0]
    context = pos_samples[1]
    pos_loss_list = [math.log(sigmoid(np.dot(word_embedding[word], context_embedding[c]))) for c in context]
    neg_loss_list = [((train_counts[c]**(3/4))/total_count)*math.log(sigmoid(-1*np.dot(word_embedding[word], context_embedding[c]))) for c in neg_samples]

    loss = sum(pos_loss_list) + sum(neg_loss_list)
    return loss, sum(pos_loss_list), sum(neg_loss_list)

def get_grad(pos_samples, neg_samples, word_embedding, context_embedding):
    word = pos_samples[0]
    context = pos_samples[1]

    word_grad = [(1-sigmoid(np.dot(word_embedding[word], context_embedding[c])))*context_embedding[c] for c in context]

    context_grad = defaultdict(list)
    [context_grad[c].append((1-sigmoid(np.dot(word_embedding[word], context_embedding[c])))*word_embedding[word]) for c in context]
    neg_context_grad = defaultdict(list)
    [neg_context_grad[c].append(-1*(1-sigmoid(np.dot(-word_embedding[word], context_embedding[c])))*word_embedding[word]) for c in neg_samples]
    neg_word_grad = defaultdict(list)
    [neg_word_grad[c].append(-1*(1-sigmoid(np.dot(-word_embedding[word], context_embedding[c])))*context_embedding[c]) for c in neg_samples]


    return word_grad, context_grad, neg_context_grad, neg_word_grad, word, context

def sgd_step(embedding, grad, lr):
    pass

def train_word2vec(tokenized_data_file, nb_neg_samples, lr, context, nb_epochs, dim, anneal):

    train_counts = pickle.load(open('dict_count_train_lemmatized.pkl','rb'))
    normalized_counts, total_count = normalize_counts(train_counts)
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
        running_loss = 0
        running_loss_p = 0
        running_loss_n = 0
        count = 0
        print('Epoch {}/{}'.format(epoch, nb_epochs - 1))
        print('-' * 10)
        loader = tqdm(range(data_size), unit='words')
        for i in enumerate(loader):
            p_samples = next(positive_samples_generator)

            if p_samples != None:
                word = p_samples[0]
                context = p_samples[1]
                for c in context:
                    negative_samples = next(negative_samples_generator)
                    positive_samples = [word, [c]]

                    loss, p_loss, n_loss = get_loss(positive_samples, negative_samples, word_embedding, context_embedding, train_counts, total_count)
                    word_grad, context_grad, neg_context_grad, neg_word_grad, word, context = get_grad(positive_samples, negative_samples, word_embedding, context_embedding)
                    if anneal:
                       if (epoch+1) %8 == 0:
                           lr = 0.5*lr
                    for w_grad in word_grad:
                        word_embedding[word] = word_embedding[word] + lr*w_grad
                    for key, w_grad in neg_word_grad.items():
                        # word_embedding[word] = word_embedding[word] + lr*(w_grad[0]*((train_counts[key]**(3/4))/total_count))
                        word_embedding[word] = word_embedding[word] + lr*w_grad[0]
                    for key, c_grad in context_grad.items():
                        # print(c_grad[0].shape)
                        context_embedding[key] = context_embedding[key] + lr*c_grad[0]
                    for key, n_grad in neg_context_grad.items():
                        context_embedding[key] = context_embedding[key] + lr*n_grad[0]
                        # context_embedding[key] = context_embedding[key] + lr*(n_grad[0]*((train_counts[key]**(3/4))/total_count))

                    if math.isnan(p_loss):
                        print(neg_word_grad)
                        exit()
                    running_loss_n += n_loss
                    running_loss_p += p_loss
                    running_loss += loss
                    count += 1

                    loader.set_postfix(w_norm=np.linalg.norm(word_embedding[word]), p_loss=running_loss_p/count,
                                        n_loss=running_loss_n/count, lr=lr)
    pickle.dump(word_embedding, open('checkpoints/word_embeddings_10_50dim_with_lemma_context_7_v5.1.2.pkl', 'wb'))
    pickle.dump(context_embedding, open('checkpoints/context_embedding_10_50dim_with_lemma_context_7_v5.1.2.pkl', 'wb'))

                # loss = get_loss(pos_samples, neg_samles)
                # word_grad, context_grad, neg_context_grad, word, context = get_grad(positive_samples, negative_samples, word_embedding, context_embedding)
def calculate_cosine_similarity(vector_1, vector_2):
    similarity = np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
    return similarity

def calculate_pearson_cofficient(simlex_file, word_embedding='checkpoints/word_embeddings_10_50dim_with_lemma_v5.1.2.pkl',
                                context_embedding='checkpoints/context_embedding_10_50dim_with_lemma_v5.1.2.pkl'):
    sim_file = open(simlex_file, 'r')
    w_embedding = pickle.load(open(word_embedding, 'rb'))
    c_embedding = pickle.load(open(context_embedding, 'rb'))
    simlex_scores = []
    model_scores = []
    i = 0
    for line in sim_file:
        i += 1
        if i > 1:
            l = line.split('\t')
            # print(l)

            try:
                # print(l)
                if args.lemmatize:
                    w1_1 = w_embedding[lemmer.lemmatize(l[0])]
                    w1_2 = c_embedding[lemmer.lemmatize(l[0])]

                    w2_1 = w_embedding[lemmer.lemmatize(l[1])]
                    w2_2 = c_embedding[lemmer.lemmatize(l[1])]
                    w1 = np.hstack([w1_1, w1_2])
                    w2 = np.hstack([w2_1, w2_2])

                    w2v_score = calculate_cosine_similarity(w1, w2)
                    simlex_scores.append(float(l[3]))
                    # print(w2v_score)
                    model_scores.append(w2v_score)
                else:
                    w1_1 = w_embedding[l[0]]
                    w1_2 = c_embedding[l[0]]

                    w2_1 = w_embedding[l[1]]
                    w2_2 = c_embedding[l[1]]
                    w1 = np.hstack([w1_1, w1_2])
                    w2 = np.hstack([w2_1, w2_2])

                    w2v_score = calculate_cosine_similarity(w1, w2)
                    simlex_scores.append(float(l[3]))
                    # print(w2v_score)
                    model_scores.append(w2v_score)
            except:
                pass

    pearson_cofficient = pearsonr(simlex_scores, model_scores)
    print(pearson_cofficient)

def get_k_NN(test_word, k=10):
    if args.lemmatize:
        word_embedding='checkpoints/word_embeddings_10_50dim_with_lemma_v5.1.2.pkl'
        context_embedding='checkpoints/context_embedding_10_50dim_with_lemma_v5.1.2.pkl'
    else:
        word_embedding='checkpoints/word_embeddings_10_50dim_vanila_with_stopwords_v5.1.1.pkl'
        context_embedding='checkpoints/context_embedding_10_50dim_vanila_with_stopwords_v5.1.1.pkl'


    word_embedding = pickle.load(open(word_embedding, 'rb'))
    context_embedding = pickle.load(open(context_embedding, 'rb'))

    if args.lemmatize:
        test_word = lemmer.lemmatize(test_word)

    score_dict = defaultdict(float)
    for key, embed in word_embedding.items():
        try:
            t_w = word_embedding[test_word]
            t_c = context_embedding[test_word]
            t = np.hstack([t_w, t_c])
        except:
            print("Word not present in the dictionary. Try a different word.")
            exit()
        w_w = embed
        w_c = context_embedding[key]
        w = np.hstack([w_w, w_c])

        similarity = calculate_cosine_similarity(t, w)
        score_dict[key] = (similarity)
    sorted_score_dict = sorted(score_dict.items(), key=operator.itemgetter(1))
    print("Test word is: %s"%(test_word))
    print(sorted_score_dict[-k:])

def syntactic_analogy():
    if args.lemmatize:
        word_embedding = 'checkpoints/word_embeddings_10_50dim_with_lemma_v5.1.2.pkl'
        context_embedding = 'checkpoints/context_embedding_10_50dim_with_lemma_v5.1.2.pkl'
    else:
        word_embedding='checkpoints/word_embeddings_10_50dim_vanila_with_stopwords_v5.1.1.pkl'
        context_embedding='checkpoints/context_embedding_10_50dim_vanila_with_stopwords_v5.1.1.pkl'
    word_embedding = pickle.load(open(word_embedding, 'rb'))
    context_embedding = pickle.load(open(context_embedding, 'rb'))

    word1 = 'quick'
    word2 = 'quickly'
    word3 = 'slow'
    if args.lemmatize:
        word1 = lemmer.lemmatize(word1)
        word2 = lemmer.lemmatize(word2)
        word3 = lemmer.lemmatize(word3)
    word1 = np.hstack([word_embedding[word1], context_embedding[word1]])
    word2 = np.hstack([word_embedding[word2], context_embedding[word2]])
    word3 = np.hstack([word_embedding[word3], context_embedding[word3]])
    word4 = word1 - word2 + word3
    score_dict = defaultdict(float)
    for key, embed in word_embedding.items():
        w_w = embed
        w_c = context_embedding[key]
        w = np.hstack([w_w, w_c])
        similarity = calculate_cosine_similarity(word4, w)
        score_dict[key] = (similarity)
    sorted_score_dict = sorted(score_dict.items(), key=operator.itemgetter(1))
    k = 10
    print(sorted_score_dict[-k:])





if args.tokenize:
    tokenize(lemmatize=args.lemmatize, stemming=args.stemming, remove_stop_words=args.remove_stop_words)

if args.build_dict:
    build_dict(args.tokenized_data_file)
if args.initilize:
    initilize_word_embeddings(args.embedding_dim)

if args.train_embeddings:
    train_word2vec(args.tokenized_data_file, args.nb_negative_samples, args.lr,
    args.context, args.nb_epochs, args.embedding_dim, args.lr_annealing)

if args.pearson_cofficient:
    calculate_pearson_cofficient(args.simlex_file)

if args.k_NN:
    get_k_NN(args.k_NN, k=args.k)

if args.syntactic_analogy:
    syntactic_analogy()
