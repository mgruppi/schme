# SChME - Semantic Change with Model Ensemble


### What is SChME
This is the semantic change detection model as seen at SemEval-2020 Task 1:
Unsupervised Detection of Lexical Semantic Change.

It combines alignment of word embeddings with voting mechanisms to quantify the semantic change between two given corpora.

### How to use

You need to provide data from a pair of sources, which can be done in two ways.

1. A pair of text files, each containing one corpora relative to different time periods or domains.
2. A pair of word embeddings in [Word2Vec format](https://radimrehurek.com/gensim/scripts/glove2word2vec.html).

Word2Vec format:

```
n (no. of words) d (dimension)
<word1> x11 x12 x13 ... x1d
<word2> x21 x22 x23 ... x2d
...
<wordn> xn1 xn2 xn3 ... xnd
```
