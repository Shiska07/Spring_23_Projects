# Name: Shiska Raut
# ID: 1001526329

# import libraries
import math
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# create a dict to store all docs, filename is the key
filenames = []
docs = []

# read and store files
corpusroot = './US_Inaugural_Addresses_s'
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        filenames.append(filename)
        doc = file.read()
        file.close() 
        doc = doc.lower()
        docs.append(doc)


# returns a list of tokens after performing tokenization, stopword removal and stemming
def get_tokens(qstring):
    
    # tokenize vocab string on spaces, punctuation, etc.
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(qstring)
    
    # remove stopwords
    tokens_stp = [word for word in tokens if word not in stopwords.words('english')]
    
    # perform stemming
    stemmer = PorterStemmer()
    tokens_stp_stm = [stemmer.stem(token) for token in tokens_stp]
    
    return tokens_stp_stm


# creates a vocabulary list using collection and search query
def get_vocab(collection):
    vocab_str = ''.join(collection)
    
    # tokenize vocab string
    all_vocab_tokens = get_tokens(vocab_str)
    unique_vocab_tokens = list(set(all_vocab_tokens))

    return unique_vocab_tokens


# create vocabulary list
vocab_list = get_vocab(docs)
print(f'The vocabulary list consists of {len(vocab_list)} items.\n')

# returns tf vector for a given document/query string
def get_tf_count_vector(str_tokens):
        
    # stores tf values for a the given string
    str_tf_vec = []
    for token in vocab_list:
        count = str_tokens.count(token)
        str_tf_vec.append(count)

    return str_tf_vec


'''
DOCUMENT TOKENIZATION
Convert all document strings into tokens for df counting.
'tokens_for_each_file' is a dictionary that contains tokens for all individual files(key)
'''
tokens_for_each_file = {}
for i, doc in enumerate(docs):
    tokens_for_each_file[filenames[i]] = get_tokens(doc)


'''
RAW DOCUMENT FREQUENCY(df) CALCULATION
'raw_df' is a dictionary that stores raw df of each token
in the vocabulary. Document frequency is specific to a certain term
'''
raw_dfs = {}
for token in vocab_list:
    raw_dfs[token] = 0
    for k, v in tokens_for_each_file.items():
        if token in v:
            raw_dfs[token] = raw_dfs[token] + 1


'''
RAW TERM FREQUENCY CALCULATIOM
'raw_tf' is a dictionary containing count vector for each doc.
Term frequency is specific to a specific term for a specific doc.
'''
raw_tfs = {}
for filename, token_list in tokens_for_each_file.items():
    raw_tfs[filename] = get_tf_count_vector(token_list)


'''
VECTOR SPACE MODEL: lnc.ltc(ddd.qqq) weighing scheme
1) weighted tf for both docs and query
2) raw df for docs, weighted df for query
3) cosine normalization for both docs and query
'''

# calculate weighted_df for docs
weighted_tfs = {}
for filename, tf_vector in raw_tfs.items():

    # get a list of weighted tf for each file
    w_tf = []
    for raw_tf in tf_vector:
        if raw_tf > 0:
            val = float(1 + math.log(raw_tf))
        else: 
            val = 0
        w_tf.append(val) 
    weighted_tfs[filename] = w_tf


# calculate tfidf for docs
tfidf_docs = {}
for filename, tf_vector in weighted_tfs.items():
    tfidf_vec = []
    for token, idf in raw_dfs.items():  # idf values
        for tf in tf_vector:            # tf value
            tfidf_vec.append(tf*idf)
    tfidf_docs[filename] = tfidf_vec


# normalizes components of a list
def get_normalized_vec(tfidf_vec):

    tfidf_sq = [n**2 for n in tfidf_vec]    # square each item
    sum_of_sq = sum(tfidf_sq)               # sum of squares
    sqrt_sum_of_sq = math.sqrt(sum_of_sq)   # normalizing factor

    norm_tfidf_vec = [(tfidf/sqrt_sum_of_sq) for tfidf in tfidf_vec]

    return norm_tfidf_vec


# calculated the dot product of two vectors
def get_cosine_similarity(v1, v2):

    n = len(v1)         # total number of items
    sim_val = 0         # stores dot product
    for i in range(n):
        sim_val = sim_val + (v1[i]*v2[i])

    return sim_val


# weighted idf for query tokens
def getidf(token):
    
    # N = total no. of documents in collection
    N = len(docs)

    if token in vocab_list:
        # calculate weighted document frequency
        w_idf = math.log(N/raw_dfs[token])
    else:
        w_idf = -1

    return w_idf

# create a dictionary containing weighted idfs for each token
weighted_idfs = {}
for token in vocab_list:
    weighted_idfs[token] = getidf(token)

# returns the tf-tdf weight of a specific token w.r.t a specific document
def getweight(filename, token):

    # create a list to store tfidf values
    tfidf_val = 0
    if filename in filenames:
        tf_vec = weighted_tfs[filename]
        token_idx = vocab_list.index(token)
        idf_val = raw_dfs[token]
        tfidf_val = tf_vec[token_idx]*idf_val

    return tfidf_val

# returns the name and score of the highest matching document
def query(qstring):

    # lowercase all query letters
    qstring = qstring.lower()
    qstring_tokens = get_tokens(qstring)

    # initialize dict to store similarity values
    cosine_sim = {}

    for filename in filenames:
        tfidf_vec = []


qstring = 'The Confederation which was early felt to be necessary was prepared from the models of the Batavian and Helvetic confederacies, the only examples which remain with any detail and precision in history, and certainly the only ones which the people at large had ever considered'

query(qstring)
