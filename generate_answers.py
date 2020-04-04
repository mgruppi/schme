import os
import numpy as np
from scipy.stats import spearmanr



# Write scores/classes as a tab separated text file
def write_answer(path, targets, scores):
    with open(path, 'w', encoding='utf-8') as f_out:
        for target in targets:
            f_out.write('\t'.join((target, str(scores[target])+'\n')))


def read_feature_file(path):
    with open(path, encoding="utf-8") as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
    words, x = zip(*data)
    return words, np.array(x, dtype=float)


# Read ground truth values
def read_answers(path="test_data_public/"):
    languages = ["english", "german", "latin", "swedish"]

    ans_bin = dict()
    ans_rank = dict()

    for language in languages:
        with open(os.path.join(path, language, "truth", "binary.txt"), encoding="utf-8") as fin:
            # Read truth files
            ans_bin[language] = {w: int(y) for w, y in map(lambda s: s.strip().split("\t", 1), fin.readlines())}
        with open(os.path.join(path, language, "truth", "graded.txt"), encoding="utf-8") as fin:
            ans_rank[language] = {w: float(y) for w, y in map(lambda s: s.strip().split("\t", 1), fin.readlines())}
    return ans_bin, ans_rank


# Load list of target words and return it as a list
def load_targets(path):
    with open(path, 'r', encoding='utf-8') as f_in:
            targets = [line.strip() for line in f_in]
    return targets


# Given a dict of answers and a dict of truth, compute evaluation
def eval_binary(answer, truth):
    acc = 0

    # Compute accuracy
    # Also compute true positives, false positives and true negatives
    # These values will be used to compute precision and recall
    tp = 0; fp = 0; fn =0; tn=0
    for word in answer:
        if answer[word] == truth[word]:
            acc += 1
            # Decide if it's true positive
            if answer[word] == 1:
                tp += 1
            else:
                tn += 1
        else:  # got a false, decide if it's positive or negative
            if answer[word] == 1:  # false positive
                fp += 1
            else:
                fn += 1

    acc = acc/len(answer)
    precision = tp/(tp+fp+1e-10)
    recall = tp/(tp+fn+1e-10)
    fallout = fp/(fp+tn+1e-10)
    f1 = 2*(precision*recall)/(precision+recall+1e-10)

    return acc, precision, recall, f1, fallout

# Evaluate ranking with spearman's rho
def eval_ranking(answer, truth):
    words = sorted(list(answer.keys()))
    x_ans = [answer[w] for w in words]
    x_tru = [truth[w] for w in words]
    r, p = spearmanr(x_ans, x_tru)

    return r



# p_features: dictionary of str-> array of feature columns
# target_ids: id (index) of words to be evaluated
# use: columns to use, if None then use all columns
def soft_voting(p_features, targets, target_ids, use=None, threshold=0.5):
    voting = dict()
    labels = dict()
    if not use:
        use = list(p_features.keys())
    for t, tid in zip(targets, target_ids):
        voting[t] = sum(p_features[f][tid] for f in use)/len(use)
        labels[t] = int(voting[t] > threshold)
    return voting, labels


# Estimate a CDF for feature distribution x
# One way this can be done is via sorting arguments according to values,
# getting a sorted array of positions (low to high)
# then normalize this by len(x)
def get_feature_cdf(x):
    y = np.argsort(x)
    p = np.zeros(len(x))
    for i, v in enumerate(y):
        p[v] = i
    p = p/len(x)
    return p


# Load features from feature/ directory and produce answers
def main():
    languages = ["english", "german", "latin", "swedish"]

    # Read ground truth
    truth_bin, truth_rank = read_answers()

    # Defines the model parameters for each language

    # Features to use in final model
    lang_features = \
    {
        "english": ["cosine"],
        "german": ["cosine", "freq_diff"],
        "latin": ["cosine", "freq_diff"],
        "swedish": ["cosine"]
    }

    # Set up thresholds for each language
    lang_t = \
    {
        "english": 0.8,
        "german": 0.7,
        "latin": 0.2,
        "swedish": 0.75
    }

    overall_acc = 0
    overall_rank = 0

    for language in languages:

        # Get features and threshold for current language
        use_features = lang_features[language]
        t = lang_t[language]

        feature_path = "features/%s/" % language
        features = dict()

        for root, dirs, files in os.walk(feature_path):
            for f in files:
                feature_name = f.split(".", 1)[0]
                words, x = read_feature_file(os.path.join(feature_path, f))
                features[feature_name] = x

        # Read target words
        word_id = {w: i for i, w in enumerate(words)}
        target_path = "test_data_public/%s/targets.txt" % language
        targets = load_targets(target_path)
        target_ids = [word_id[t] for t in targets]

        # Compute all CDFs
        p_features = {f: get_feature_cdf(features[f]) for f in features}


        # Get votes and aggregate votes into final class decision
        # based on threshold t
        voting, classes = soft_voting(p_features, targets, target_ids,
                                      use=use_features, threshold=t)


        # Write answer files
        out_task1 = "answer/task1/%s.txt" % language
        out_task2 = "answer/task2/%s.txt" % language

        if not os.path.exists("answer"):
            os.mkdir("answer")
        if not os.path.exists("answer/task1"):
            os.mkdir("answer/task1")
        if not os.path.exists("answer/task2"):
            os.mkdir("answer/task2")

        # Get accuracy, precision, and recall
        acc_bin, prec_bin, rec_bin, f1_bin, fo_bin = eval_binary(classes, truth_bin[language])
        # Get spearman rho for ranking
        rank_r = eval_ranking(voting, truth_rank[language])

        overall_acc += acc_bin
        overall_rank += rank_r


        print("%s\t| t=%.2f\t| acc=%.3f\t| rank=%.3f"
               % (language, t, acc_bin, rank_r))

        write_answer(out_task1, targets, classes)
        write_answer(out_task2, targets, voting)

    print("-"*30)
    print("Overall\t| acc=%.3f\t| rank=%.3f" % (overall_acc/4, overall_rank/4))



if __name__ == "__main__":
    main()
