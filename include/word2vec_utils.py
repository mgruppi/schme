from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
from scipy.linalg import orthogonal_procrustes
from include.WordVectors import WordVectors
import numpy as np
import time
from include.utils import log_time


# Functions to handle word embedding, learning, loading, etc


# Converts Word2Vec model into WordVectors class
# We do not need the whole trainable Word2Vec model
# We'll only use the learned vectors
def w2v_to_wv(model):
    words = [w for w in model.wv.vocab]
    vectors = [model.wv[w] for w in words]
    wv = WordVectors(words=words, vectors=vectors)

    return wv



# Train word2vec model from input path
def train_word2vec(sentences,
                   dim=300, window=10, workers=12, min_count=10,
                   sorted_vocab=1):

    t0 = time.time()

    print("Training Word2Vec")
    model = Word2Vec(sentences, size=dim, window=window,
                     min_count=min_count,
                     sorted_vocab=sorted_vocab,
                     workers=12)

    log_time(t0)

    # Convert Word2Vec model to WordVectors and return
    return w2v_to_wv(model)


# Input:    wv - list of WordVectors
#           landmarks - Number of landmarks to use. If none, then the every
#                       word is used
#           random_landmarks -  toggle use of random words for landmarks
#           target          -   (int) index of WordVector to be used as target
#                               default 0: align on the first source
#                               if -1 or "mean"
#                                   use the mean point for every word
# Return - list of aligned WordVectors [, landmarks (optionally)]
def align_wordvectors(*wv, landmarks=None, random_landmarks=False, target=0,
                      return_intersect=False, least_freq=False,
                      return_loss=False,
                      exclude=set()):
    if len(wv) < 2:
        print("! Error: not enough WordVectors for alignment (2 required)")
        return None

    # Computes the mean vectors for every intersecting word in the input
    # The resulting WordVector will become the target for the alignment
    if target == -1 or target == "mean":
        tgt_words = list(set.intersection(*[set(s.words) for s in wv]))
        tgt_vecs = list()
        for w_ in wv:
            tgt_vecs.append([w_[word] for word in tgt_words])
        tgt_vecs = np.sum(tgt_vecs, 0)/len(wv)
        tgt_wv = WordVectors(words=tgt_words, vectors=tgt_vecs)
    else:
        tgt_wv = wv[target]  # otherwise set target from input

    aligned_wv = list()
    isect_list = list()

    # If landmarks is not provided, use all possible words
    if not landmarks:
        landmarks = tgt_wv.words
    # If int, then select first n landmarks
    elif isinstance(landmarks, int):
        if least_freq:
            landmarks = tgt_wv.words[-landmarks:]
        else:
            landmarks = tgt_wv.words[:landmarks]
    elif isinstance(landmarks, list):
        landmarks = [tgt_wv.words[i] for i in landmarks]


    # We can align multiple wordvectors at once (not just 2)
    for src_wv in wv[0:]:
        if not random_landmarks:
            # Get the intersection between current source and
            # the landmark set of words
            isect = [w for w in src_wv.words if w in landmarks]

        else:  # WARNING: for random_landmarks, landmakrs must be int
            isect = set.intersection(set(src_wv.words),
                                     set(tgt_wv.words))
            isect = isect - exclude
            isect = np.random.choice(list(isect), size=len(landmarks))

        # Get target vectors for selected words
        isect_tgt = np.array([tgt_wv[w] for w in isect])
        isect_src = np.array([src_wv[w] for w in isect])

        # Perform procrustes alignment
        r, scale = orthogonal_procrustes(isect_src, isect_tgt)

        # Given transformation matrix r, we apply it to src_wv via matrix mult
        a_wv = WordVectors(src_wv.words, np.dot(src_wv.vectors, r))
        aligned_wv.append(a_wv)
        isect_list.append(isect)

    if return_intersect:
        return aligned_wv, isect
    else:
        return aligned_wv
