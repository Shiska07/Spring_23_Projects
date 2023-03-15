# Name: Shiska Raut
# ID: 1001526329

# import libraries
import math
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# create a list to store all filenames and document content
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


# create vocabulary list using the corpus
vocab_list = get_vocab(docs)


# returns tf vector for a given document/query string using the vocabulary
def get_tf_count_vector(tokens_list):
        
    # stores tf values of tokens in tokens_List
    raw_tf_vec = []
    for token in vocab_list:
        count = tokens_list.count(token)
        raw_tf_vec.append(count)

    # return a list containing raw tf values
    return raw_tf_vec


'''
DOCUMENT TOKENIZATION
Convert all document strings into tokens for df counting.
'tokens_for_each_file' is a dictionary that contains tokens for all individual files(key)
'''
tokens_for_each_doc = {}
for i, doc in enumerate(docs):
    tokens_for_each_doc[filenames[i]] = get_tokens(doc)


'''
RAW DOCUMENT FREQUENCY(df) CALCULATION
'raw_df' is a dictionary that stores raw df of each token
in the vocabulary. Document frequency is specific to a certain term
'''
raw_dfs = {}

# for each token in the corpus vocab
for token in vocab_list:

    # set token count to '0' initially
    raw_dfs[token] = 0
    for k, v in tokens_for_each_doc.items():

        # of token occurs in a doc, increase count
        if token in v:
            raw_dfs[token] = raw_dfs[token] + 1


'''
RAW TERM FREQUENCY(tf) CALCULATIOM
'raw_tf' is a dictionary containing count vector for each doc.
Term frequency is specific to a specific term for a specific doc.
'''
docs_raw_tfs = {}
for filename, token_list in tokens_for_each_doc.items():
    docs_raw_tfs[filename] = get_tf_count_vector(token_list)


'''
VECTOR SPACE MODEL: lnc.ltc(ddd.qqq) weighing scheme
1) weighted tf for both docs and query
2) raw df for docs, weighted idf for query
3) cosine normalization for both docs and query
'''

# returns a list of weighted tf values given a list of raw tf values
def get_weighted_tf(raw_tf_vec):

    # list to store weighted tf
    w_tf_vec = []
    for raw_tf in raw_tf_vec:
        if raw_tf > 0:
            val = float(1 + math.log(raw_tf))
        else: 
            val = 0
        w_tf_vec.append(val)
    
    return w_tf_vec


# calculate weighted_tf for docs
weighted_tfs = {}
for filename, raw_tf_vec in docs_raw_tfs.items(): 
    weighted_tfs[filename] = get_weighted_tf(raw_tf_vec)


# returns a list of tfidf vvalues givrn given a list of weighted tf and idf values
def get_tfidf(tf_vec, idf_vec):

    # initialize list to store tfidf values
    tfidf_vec = []
    n = len(tf_vec)
    for i in range(n):
        tfidf_vec.append(tf_vec[i]*idf_vec[i])

    return tfidf_vec


# normalizes components of a list
def get_normalized_vec(tfidf_vec):

    tfidf_sq = [n**2 for n in tfidf_vec]    # square each item
    sum_of_sq = sum(tfidf_sq)               # sum of squares
    sqrt_sum_of_sq = math.sqrt(sum_of_sq)   # normalizing factor

    norm_tfidf_vec = [(tfidf/sqrt_sum_of_sq) for tfidf in tfidf_vec]

    return norm_tfidf_vec


# calculate tfidf values for all docs
tfidf_docs = {}
for filename, w_tf_vec in weighted_tfs.items():

    # using 'lnc' scheme for 'ddd', we use weighted tf and raw df values
    # for caluclating tf*idf
    tfidf_vec = get_tfidf(w_tf_vec, list(raw_dfs.values()))

    # store normalized tfidf vector
    tfidf_docs[filename] = get_normalized_vec(tfidf_vec)


# retuns the dot product/cosine similarity of items in two lists
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

    tfidf_val = float(0)
    if filename in filenames:
        if token in vocab_list:

            # get index of the given token from vocab_list
            token_idx = vocab_list.index(token)

            # get tfidf value of token from 'tfidf_docs' dict containing
            # tfidf values for all docs
            tfidf_val = tfidf_docs[filename][token_idx]

    return tfidf_val


# returns the name and score of the highest matching document
def query(qstring):

    # lowercase all query letters
    qstring = qstring.lower()
    qstring_tokens = get_tokens(qstring)

    # initialize dict to store similarity values
    cosine_sim = {}

    # get weighted tf vector for query
    raw_tf_query = get_tf_count_vector(qstring_tokens)
    w_tf_query = get_weighted_tf(raw_tf_query)

    # as per the 'ltc'weighing scheme for 'qqq', we use weighted idf values
    w_idfs = list(weighted_idfs.values())

    # get tfidf values
    tfidf_query = get_tfidf(w_tf_query, w_idfs)

    # normalize values
    tfidf_query_norm = get_normalized_vec(tfidf_query)
    
    # caluclate cosine similarity with each doc
    for filename in filenames:

        # calculate cosine similarity of query with all docs
        cosine_sim[filename] = get_cosine_similarity(tfidf_docs[filename], tfidf_query_norm)

    # get filename with maximum similarity
    max_sim_fname = ''
    max_sim_val = float(0)
    for fname, sim_val in cosine_sim.items():
        if sim_val > max_sim_val:
            max_sim_fname = fname
            max_sim_val = sim_val

    return (max_sim_fname, max_sim_val)


qstring = 'Having thus imparted to you my sentiments as they have been awakened by the occasion which brings us together, I shall take my present leave; but not without resorting once more to the benign Parent of the Human Race in humble supplication that, since He has been pleased to favor the American people with opportunities for deliberating in perfect tranquillity'

print(query(qstring))
