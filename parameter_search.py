# Apply grid search for find best combination of landmark and threshold
# for the problem of classification

import os
import numpy as np
from generate_answers import read_answers, eval_binary, eval_ranking, \
                             write_answer, load_targets, get_feature_cdf
from collections import Counter



def read_feature_file(path):
    with open(path, encoding="utf-8") as fin:
        data = map(lambda s: s.strip().split("\t"), fin.readlines())
    words, x = zip(*data)
    return words, np.array(x, dtype=float)


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


def main():
    languages = ["english", "german", "latin", "swedish"]
    linestyles = {"english": "solid", "german": (0, (3, 1, 1, 1, 1, 1)),
                  "latin": "dashed",
                  "swedish": "dashdot"}

    linecolors = {"english": "#003f5c",  "german": "#7a5195",
                  "latin": "#ef5675", "swedish": "#ffa600"}
    truth_bin, truth_rank = read_answers()  # read ground truth

    feature_configs = [["cosine"], ["map"], ["cosine", "freq_diff"],
                        ["map", "freq_diff"],
                        ["cosine", "freq_diff", "map"]]

    thres = np.arange(0.0, 1.0, 0.05)
    thres = [round(t, 2) for t in thres]

    total_best_acc = 0
    total_best_rank = 0

    # Begin language
    for language in languages:
        feature_path = "features/landmarks/%s" % language

        x_landmarks = list()  # store list of int landmarks

        # Make list of items
        items = [i for i in os.listdir(feature_path) if os.path.isdir(os.path.join(feature_path, i))]
        items = sorted([int(i) for i in items])  # sort paths
        items = items[1:] + [items[0]]  # quickfix to show all landmarks at the end

        # Print class ratio
        labels = Counter(truth_bin[language].values())
        print(language)
        print("Majority", labels, round(max(labels.values())/sum(labels.values()), 3))
        # Store best parameters and values
        best_acc = 0
        best_land = -1
        best_t = -1
        best_feature = []
        best_rank = 0
        best_land_rank = -1
        best_feature_rank = []
        best_ans_cls = None
        best_ans_rank = None
        # Stores accuracy grid thresholds X landmarks
        acc_grid = np.zeros((len(thres), len(items)))
        # Begin landmarks
        for li, item in enumerate(items):
            if not os.path.isdir(os.path.join(feature_path, str(item))):
                continue
            f_path = os.path.join(feature_path, str(item))
            features = dict()

            for root, dirs, files in os.walk(f_path):
                for f in files:
                    feature_name = f.split(".", 1)[0]
                    words, x = read_feature_file(os.path.join(f_path, f))
                    features[feature_name] = x

            word_id = {w: i for i, w in enumerate(words)}
            target_path = "test_data_public/%s/targets.txt" % language
            targets = load_targets(target_path)
            target_ids = [word_id[t] for t in targets]


            # If landmark value is 'None', this means we use all words as landm
            if item == -1:
                x_landmarks.append(len(word_id))
            else:
                x_landmarks.append(int(item))

            # Compute all CDFs
            p_features = {f: get_feature_cdf(features[f]) for f in features}

            # Begin threshold
            for ti, t  in enumerate(thres):
                for use_features in feature_configs:
                    voting, classes = soft_voting(p_features, targets, target_ids,
                                                  use=use_features, threshold=t)

                    # Get accuracy, precision, and recall
                    acc_bin, prec_bin, rec_bin, f1_bin, fo_bin = eval_binary(classes, truth_bin[language])
                    r_rank = eval_ranking(voting, truth_rank[language])
                    acc_grid[ti][li] = acc_bin
                    if acc_bin > best_acc or (acc_bin == best_acc and x_landmarks[-1] < best_land):
                        best_acc = round(acc_bin, 3)
                        best_t = t
                        best_land = x_landmarks[-1]
                        best_feature = use_features
                        best_ans_cls = classes

                    if r_rank > best_rank or (r_rank == best_rank and x_landmarks[-1] < best_land_rank):
                        best_rank = round(r_rank, 3)
                        best_land_rank = x_landmarks[-1]
                        best_feature_rank = use_features
                        best_ans_rank = voting

        print(language)
        print("- Best cls", best_acc, best_land, best_t, best_feature)
        print("     - acc:", best_acc)
        print("     - landmarks:", best_land, round(best_land/len(words),2 ))
        print("     - t:", best_t)
        print("     - best_feature:", best_feature)

        print("- Best ranking")
        print("     - acc:", best_rank)
        print("     - landmarks:", best_land_rank, round(best_land_rank/len(words), 2))
        print("     - best_feature:", best_feature_rank)
        print()

        # Save best answers
        out_task1 = "answer/task1/%s.txt" % language
        out_task2 = "answer/task2/%s.txt" % language

        if not os.path.exists("answer"):
            os.mkdir("answer")
        if not os.path.exists("answer/task1"):
            os.mkdir("answer/task1")
        if not os.path.exists("answer/task2"):
            os.mkdir("answer/task2")

        write_answer(out_task1, targets, best_ans_cls)
        write_answer(out_task2, targets, best_ans_rank)

        total_best_acc += best_acc
        total_best_rank += best_rank

    print("+ Final best acc", total_best_acc/4)
    print("+ Final best rank", total_best_rank/4)


if __name__ == "__main__":
    main()
