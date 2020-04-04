import argparse
import os
import time
from gensim.models.word2vec import PathLineSentences
from include import extract_features
from include.WordVectors import WordVectors, intersection
from include.utils import log_time
from include import word2vec_utils
from include.word2vec_utils import align_wordvectors


# Run on input data (raw corpora or word vector pair)



# Reproduce experiments from SemEval-2020 Task 1
def main():

    path = "test_data_public"

    if not os.path.exists(path):
        print("Could not find path", path)
        print("Please run download-semeval-data.sh to download the data")
        return 0

    languages = ["english", "german", "latin", "swedish"]


    for lang in languages:

        print("Language:", lang)

        print(" - Loading corpora")
        path_corpus1 = "test_data_public/%s/corpus1/lemma/" % lang
        path_corpus2 = "test_data_public/%s/corpus2/lemma/" % lang
        corpus1 = PathLineSentences(path_corpus1)
        corpus2 = PathLineSentences(path_corpus2)

        # First, check if we can find the word vectors for each language
        # (if it has already been computed, we don't have to do it again)
        if not os.path.exists("wordvectors"):
            os.mkdir("wordvectors")
        if not os.path.exists("wordvectors/%s" % lang):
            os.mkdir("wordvectors/%s" % lang)

        # Check word vectors for corpus1
        # Save word vectors for later use
        print(" - Checking wordvectors")
        # Check if we need to train word2vec
        if not os.path.exists("wordvectors/%s/1.vec" % lang):
            wv1 = word2vec_utils.train_word2vec(corpus1)
            wv1.save_txt("wordvectors/%s/1.vec" % lang)
        else:  # already trained, load word vectors
            wv1 = WordVectors(input_file="wordvectors/%s/1.vec" % lang)
        if not os.path.exists("wordvectors/%s/2.vec" % lang):
            wv2 = word2vec_utils.train_word2vec(corpus2)
            wv2.save_txt("wordvectors/%s/2.vec" % lang)
        else:
            wv2 = WordVectors(input_file="wordvectors/%s/2.vec" % lang)

        # Compute intersecting words and align wordvectors
        wv1, wv2 = intersection(wv1, wv2)
        wv1, wv2 = align_wordvectors(wv1, wv2)

        words = wv1.words  # both wv1 and wv2 now have the same words

        # Extract all features
        if not os.path.exists("features"):
            os.mkdir("features")
        # Create file directory for current language, if needed
        if not os.path.exists("features/%s/" % lang):
            os.mkdir("features/%s/" % lang)

        features = dict()  # store features here
        t0 = time.time()

        # Searches for pre-computed features to avoid computing them again
        # Only re-compute if feature file is not found
        # In particular, search for file features/<language>/<feature>.csv

        # Check for cosine
        f_path = "features/%s/cosine.csv" % lang
        if not os.path.exists(f_path):
            print("Computing cosine")
            features["cosine"] = extract_features.get_metric(wv1, wv2,
                                                             words, "cosine")
            extract_features.save_feature_file(f_path,
                                                features["cosine"],
                                                words)
        # load cosine.csv file
        print("Loading feature file", f_path)
        features["cosine"] = extract_features.read_feature_file(f_path)

        # Check frequency differential
        f_path = "features/%s/freq_diff.csv" % lang
        if not os.path.exists(f_path):
            print("Computing freq diff")
            dico = extract_features.frequency_differential_scores(corpus1, corpus2, words)
            extract_features.save_feature_file(f_path,dico, words)
        # load freq_diff.csv
        print("Loading feature file", f_path)
        features["freq_diff"] = extract_features.read_feature_file(f_path)

        # Check for MAP feature
        f_path = "features/%s/map.csv" % lang
        if not os.path.exists(f_path):
            print("Computing map")
            dico = extract_features.map_distance(wv1, wv2, words)
            extract_features.save_feature_file(f_path, dico, words)
# load map.csv
        print("Loading feature file", f_path)
        features["map"] = extract_features.read_feature_file(f_path)

        log_time(t0)


if __name__ == "__main__":
    main()
