# import libraries
import math
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# create a list to store all filenames and document content
docs = {}

# read and store files
corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        docs[filename] = file.read().lower()
        file.close()

# returns a list of tokens after performing tokenization, stopword removal and stemming
def get_tokens(qstring):
    # tokenize vocab string on spaces, punctuation, etc.
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(qstring)

    # remove stopwords
    stopwords_list = stopwords.words('english')
    tokens_stp = [word for word in tokens if word not in stopwords_list]

    # perform stemming
    stemmer = PorterStemmer()
    tokens_stp_stm = [stemmer.stem(tk) for tk in tokens_stp]

    # return final list of tokens with stemmed tokens and stopwords removed
    return tokens_stp_stm

# creates a vocabulary list using collections/doc corpus
def get_vocab(collection):

    # create a list of document strings
    docs_list = list(collection.values())

    # the vocab string contains strings for all docs joined together
    vocab_str = ''.join(docs_list)

    # tokenize vocab string
    all_vocab_tokens = get_tokens(vocab_str)

    # filter unique tokens anc create a list
    unique_vocab_tokens = list(set(all_vocab_tokens))

    # return a list of unique tokens
    return unique_vocab_tokens

# create vocabulary list using the corpus
vocab_list = get_vocab(docs)

# returns tf count for a given document/query string using the vocabulary
def get_tf_count_vector(tokens_list):

    # stores tf values of tokens in tokens_List
    tf_count_vec = [tokens_list.count(tk) for tk in vocab_list]

    # return a list containing raw tf values
    return tf_count_vec


'''
DOCUMENT TOKENIZATION
Convert all document strings into tokens for vocaab tokens df counting.
'tokens_for_each_file' is a dictionary that contains tokens for all individual files(key)
'''
tokens_for_each_doc = {}
for filename, doc in docs.items():
    tokens_for_each_doc[filename] = get_tokens(doc)

'''
RAW DOCUMENT FREQUENCY(df) CALCULATION for vocab tokens
'raw_df' is a dictionary that stores raw df of each token
in the vocabulary. Document frequency(df)is specific to a certain token.
'''
tokens_raw_df = {}
# for each token in the corpus vocab
for token in vocab_list:

    # set token count to '0' initially
    tokens_raw_df[token] = 0
    for filename, doc_tokens_list in tokens_for_each_doc.items():

        # of token occurs in a doc, increase count
        if token in doc_tokens_list:
            tokens_raw_df[token] = tokens_raw_df[token] + 1

'''
RAW TERM FREQUENCY(tf) CALCULATION for all docs
'docs_raw_tfs' is a dictionary containing count vector for each doc.
Term frequency is specific to a specific term for a specific doc.
'''
docs_raw_tfs = {}
for filename, doc_tokens_list in tokens_for_each_doc.items():
    docs_raw_tfs[filename] = get_tf_count_vector(doc_tokens_list)


# returns a list of tfidf values given a list of weighted tf and idf values
def get_tfidf_vec(tf_vec, idf_vec):

    # initialize list to store tfidf values
    tfidf_vector = []
    n = len(tf_vec)
    for idx in range(n):
        tfidf_vector.append(tf_vec[idx] * idf_vec[idx])

    return tfidf_vector


# normalizes components of a list
def get_normalized_vec(vec):

    tfidf_sq = [n ** 2 for n in vec]           # square each item
    sum_of_sq = sum(tfidf_sq)                  # sum of squares
    sqrt_sum_of_sq = math.sqrt(sum_of_sq)      # normalizing factor

    # if all values in tfidf_vec are zeros, that would
    # lead to a zero division
    if sqrt_sum_of_sq != 0:
        norm_tfidf_vec = [(tfidf / sqrt_sum_of_sq) for tfidf in vec]
    else:
        norm_tfidf_vec = vec.copy()

    return norm_tfidf_vec


# returns the dot product/cosine similarity of items in two lists
def get_cosine_similarity(v1, v2):

    n = len(v1)  # total number of items
    sim_val = 0  # stores dot product

    for idx in range(n):
        # since both document and query vectors are normalized,
        # we can just multiply and add all components
        sim_val = sim_val + (v1[idx] * v2[idx])

    return sim_val


'''
VECTOR SPACE MODEL: lnc.ltc(ddd.qqq) weighing scheme
1) weighted tf for both docs and query
2) raw df for docs, weighted idf for query
3) cosine normalization for both docs and query
'''

# returns a list of weighted tf values given a list of raw tf values
def get_weighted_tf_vec(tf_vec):
    # list to store weighted tf
    weighted_tf_vec = []
    for raw_tf in tf_vec:
        if raw_tf > 0:
            val = float(1 + math.log(raw_tf, 10))
        else:
            val = 0
        weighted_tf_vec.append(val)

    return weighted_tf_vec


# calculate weighted_tf for docs
docs_weighted_tfs = {}
for filename, doc_raw_tfs_vec in docs_raw_tfs.items():
    docs_weighted_tfs[filename] = get_weighted_tf_vec(doc_raw_tfs_vec)


# weighted idf for query tokens
def getidf(tk):
    if type(tk) != str:
        tk = str(tk)

    # N = total no. of documents in collection
    N = len(docs)

    if tk in vocab_list:
        # calculate weighted document frequency
        w_idf = math.log(N / tokens_raw_df[tk], 10)
    else:
        w_idf = -1
        print(f'token "{tk}" does not exist in the corpus.\n')

    return w_idf


# create a dictionary containing weighted idfs for each token
tokens_weighted_idf = {}
for token in vocab_list:
    tokens_weighted_idf[token] = getidf(token)


# returns the name and score of the highest matching document
def query(qstring):

    if type(qstring) != str:
        qstring = str(qstring)

    # lowercase all query letters
    qstring_l = qstring.lower()
    qstring_tokens = get_tokens(qstring_l)

    # initialize dict to store similarity values
    cosine_sim = {}

    # get weighted tf vector for query
    raw_tf_query = get_tf_count_vector(qstring_tokens)
    w_tf_query = get_weighted_tf_vec(raw_tf_query)

    # as per the 'ltc' weighing scheme for 'qqq', we use weighted idf values
    w_idfs = list(tokens_weighted_idf.values())

    # get normalized tfidf vector for query
    tfidf_query_norm = get_normalized_vec(get_tfidf_vec(w_tf_query, w_idfs))

    # calculate cosine similarity with each doc
    for f_name, weighted_tfs_vec in docs_weighted_tfs.items():

        # get tfidf vector for the doc as per 'lnc' weighing scheme
        tfidf_doc_norm = get_normalized_vec(get_tfidf_vec(weighted_tfs_vec, list(tokens_raw_df.values())))

        # calculate cosine similarity of query with all docs
        cosine_sim[f_name] = get_cosine_similarity(tfidf_doc_norm, tfidf_query_norm)

    # sort documents in decreasing order of cosing similarity
    sorted_dict = sorted(cosine_sim.items(), key = lambda x:x[1], reverse = True)

    # print filenames and similarity values
    print(f'Similarity values for the given query:\n')
    for f_name, sim_val in sorted_dict:
        print(f'{f_name}: {sim_val:0.3f}')
    print('\n')

# get query string from user
qstr = str(input('Provide query string: '))

# display results
query(qstr)
