import numpy as np
import pandas as pd
import csv
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz
from sparse_dot_topn import awesome_cossim_topn


def ngrams(string: str, n=3):
    # string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode()  # remove non ascii chars
    string = string.lower()  # make lower case
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)  # remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title()  # normalise case - capital at start of each word
    string = re.sub(' +', ' ', string).strip()  # get rid of multiple spaces and replace with a single space
    string = ' ' + string + ' '  # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD', r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


print("Reading data...")
name_list: pd.DataFrame = pd.read_csv("../data/s1_drug_name_list.csv")
name_list: pd.Series = name_list.iloc[:, 0]
name_list: list = name_list.tolist()

print("Vectorizing...")
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(name_list)

print("Generating CSR matrix with matches...")
matches: csr_matrix = awesome_cossim_topn(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.85, use_threads=True, n_jobs=8)

print("Saving CSR matrix...")
save_npz(file="../output/s1_cossim_matrix.npz", matrix=matches)
