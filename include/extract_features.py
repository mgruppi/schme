import os
from collections import defaultdict
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors
import numpy as np


def save_feature_file(path, dico, words):
    with open(path, "w", encoding="utf-8") as fout:
        for w in words:
            fout.write("%s\t%f\n" % (w, dico[w]))


def read_feature_file(path):
    with open(path, encoding="utf-8") as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
    words, x = zip(*data)
    return np.array(x, dtype=float)

# Extract all features from input corpora/word vectors
# Files are saved in features/<language>/<feature-name>.csv
# The format of the output file is <word>,<value>
# Those values already measure the "shift" or change between corpora 1 and 2
# Input:
#           - corpora corpus1 and corpus2
#           - wordvectors wv1 and wv2
#           - output_path (where to save files)
# Returns dict of features feature_name->values(array)
def extract_features(corpus1, corpus2, wv1, wv2):

    # wv1 and wv2 now contain the same words in the same order
    words = wv1.words

    # Initialize dictionary of features
    features = dict()

    features["freq_diff"] = frequency_differential_scores(corpus1, corpus2, words)
    features["cosine"] = get_metric(wv1, wv2, words, "cosine")
    feature["map"] = map_distance(wv1, wv2)

# Get relative frequency for all words in corpus
# Corpus must be tokenized
# Returns a dictionary of word->frequency
def get_frequency(corpus):
    print('Counting',)
    count = defaultdict(int)
    total = 0
    for line in corpus:
        for token in line:
            count[token] += 1
            total += 1
    # Normalize count
    for word in count:
        count[word] = count[word]/total

    return count


# Get the frequency differential between words in dico1 and dico2
# words - contains the list of words to compute this feature for
# Returns list of freqquency diferentials, follows order of <words>
def frequency_differential_scores(corpus1, corpus2, words):
    x = dict()
    dico1 = get_frequency(corpus1)
    dico2 = get_frequency(corpus2)

    for i, t in enumerate(words):
        x[t] = (dico1[t] - dico2[t])/(dico1[t] + dico2[t])
    return x


# Compute any metric between wv_a and wv_b, returning a vector
def get_metric(wv_a, wv_b, words, metric, m=None):
    scores = dict()
    if metric == "cosine":
        def eval(x, y):
            return cosine(x, y)
    # For precomputed metrics, the input is a n x n matrix with pairwise
    # distances already given, we use the index of a word to obtain
    # its values
    elif metric == "precomputed":
        if not m:
            print("Error: matrix m not given for metric 'precomputed'")

        def eval(i):
            return m[i]

    else:
        def eval(x, y):
            return 'nan'

    for target in words:
        try:
            if metric != "precomputed":
                scores[target] = round(eval(wv_a[target], wv_b[target]), 4)
            else:
                scores[target] = round(eval(wv_a.word_id[target]))
        except KeyError as e:  # if words not found, report nan
            scores[target] = 'nan'

    return scores


# Generate the inverse neighborhood distance vector for given sources
# Input: wordvectors a and b, and knn neighbors (int)
# Output: vector d of inverse neighborhood distance
# *** input word vectors must have the same words in the same order
# For neighbors of w x, y, z in a and n, o, p in b, the inverse neighborhood
# distance is d = || mean(x, y, z) - mean(n, o, p) ||
def mapped_neighborhood_distance(wv_a, wv_b, list_a, list_b, knn):
    d = np.zeros((len(wv_a.words)), dtype=float)

    for i in range(len(wv_a.words)):
        # Compute mean neighborhood vectors for A and B
        mean_a = sum(wv_a[u] for u in list_a[i])/knn
        mean_b = sum(wv_b[v] for v in list_b[i])/knn
        d[i] += cosine(mean_a, mean_b)

    return d


def map_distance(wv_a, wv_b, words, knn=10):
    nbrs_a = NearestNeighbors(n_neighbors=knn, n_jobs=-1).fit(wv_a.vectors)
    nbrs_b = NearestNeighbors(n_neighbors=knn, n_jobs=-1).fit(wv_b.vectors)

    # Find nearest neighbors of a word in its own source
    dist_a, list_a = nbrs_a.kneighbors()
    dist_b, list_b = nbrs_b.kneighbors()

    map_neighborhood = mapped_neighborhood_distance(wv_a, wv_b, list_a,
                                                     list_b, knn)
    dico = dict()
    for w in words:
        dico[w] = map_neighborhood[wv_a.word_id[w]]

    return dico
