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


# Get query string
# query = str(input('Type search query:'))
query = 'The Confederation which was early felt to be necessary was prepared from the models of the Batavian and Helvetic confederacies, the only examples which remain with any detail and precision in history, and certainly the only ones which the people at large had ever considered'
query = query.lower()


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
def get_vocab(collection, query):
    vocab_str = ''.join(collection)
    vocab_str = vocab_str + query
    
    # tokenize vocab string
    all_vocab_tokens = get_tokens(vocab_str)
    unique_vocab_tokens = list(set(all_vocab_tokens))

    return unique_vocab_tokens


# create vocabulary list
vocab_list = get_vocab(docs, query)
print(f'The vocabulary list consists of {len(vocab_list)} items.\n')

# returns tf vector for a given document/query string
def get_tf_count_vector(string):
    # create tokens for each document
    str_tokens = get_tokens(string)
        
    # stores tf values for a the given string
    str_tf_vec = []
    for token in vocab_list:
        count = str_tokens.count(token)
        str_tf_vec.append(count)

    return str_tf_vec


'''
FiILE & QUERY TOKENIZATION
Convert all document strings into tokens for df counting.
'tokens_for_each_file' is a dictionary that contains tokens for all individual files(key) including the query
'''
tokens_for_each_file = {}
for i, doc in enumerate(docs):
    tokens_for_each_file[filenames[i]] = get_tokens(doc)
# append tokens for the query at the end

tokens_for_each_file['query'] = get_tokens(query)

'''
RAW DOCUMENT FREQUENCY(df) CALCULATION
'raw_df' is a dictionary that stores raw df of each token
in the vocabulary
'''
raw_df = {}
for token in vocab_list:
    raw_df[token] = 0
    for k, v in tokens_for_each_file.items():
        if token in v:
            raw_df[token] = raw_df[token] + 1


# returns inverse document frequency(idf) of a token 
# using the entire vocabulary and collection
def getidf(token):
    
    # total no. of documents in collection
    N = len(docs)


    pass

# returns the tf-tdf weight of a specific token w.r.t a specific document
def getweight(filename, token):
    pass

# returns the name and score of the highest matching document
def query(qstring):
    pass
