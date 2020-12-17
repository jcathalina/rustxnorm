import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz, load_npz
from sparse_dot_topn import awesome_cossim_topn

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


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


def generate_csr_matrix(src_path: str, dest_path: str, topn=10, lower_bound=0.85):
    print("Reading data...")
    name_list: pd.DataFrame = pd.read_csv(src_path)
    name_list: pd.Series = name_list.iloc[:, 0]
    name_list: list = name_list.tolist()

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(name_list)

    print("Generating CSR matrix with matches...")
    matches: csr_matrix = awesome_cossim_topn(tf_idf_matrix, tf_idf_matrix.transpose(), topn, lower_bound,
                                              use_threads=True, n_jobs=8)

    print("Saving CSR matrix...")
    save_npz(file=dest_path, matrix=matches)


def vectorize_data(src_path: str, dest_path: str):
    print("Reading data...")
    name_list: pd.DataFrame = pd.read_csv(src_path)
    name_list: pd.Series = name_list.iloc[:, 0]
    name_list: list = name_list.tolist()

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(name_list)

    print("Saving sparse matrix...")
    save_npz(file=dest_path, matrix=tf_idf_matrix)


def match_to_fda(dest_path: str, to_compare: str):
    print("Loading FDA dictionary vectorized data...")
    fda = load_npz("../output/fda_dict_vectorized.npz")

    print("Generating CSR matrix with matches...")

    # cossim_matrix = load_npz(to_compare)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

    with open(file="../data/s1_drug_name_list_unique.csv") as csvfile:
        for name in csvfile:
            if name.__contains__("drug_name"):
                continue
            v_name = vectorizer.fit_transform(list(name))
            best_match = awesome_cossim_topn(v_name.reshape(fda.shape), fda, 1, 0.85, use_threads=True, n_jobs=8)
            print(best_match)
            break
    # matches: csr_matrix = awesome_cossim_topn(fda, cossim_matrix, 1, 0.85, use_threads=True, n_jobs=8)
    #
    # print("Saving CSR matrix...")
    # save_npz(file=dest_path, matrix=matches)

