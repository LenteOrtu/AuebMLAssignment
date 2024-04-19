import pandas as pd
import numpy as np
from random import sample
from graph import *
from utils import *
from tqdm import tqdm
from sklearn.naive_bayes import BernoulliNB

class Training:
    def __init__(self, pos_probs, neg_probs, p_1):
        self.pos_probs = pos_probs
        self.neg_probs = neg_probs
        self.p_0 = 1 - p_1
        self.p_1 = p_1

def train_naive_bayes(dataset):
    x_train, y_train = dataset
    examples, _ = x_train.shape
    pos_counts = np.count_nonzero(y_train == 1)
    p_1 = pos_counts / examples
    pos_probs = (np.sum(x_train[np.where(y_train == 1)], axis=0) + 1) / (pos_counts + 2)
    neg_probs = (np.sum(x_train[np.where(y_train == 0)], axis=0) + 1) / (examples - pos_counts + 2)

    return Training(pos_probs, neg_probs, p_1)

def test_naive_bayes(x, training):
    p_0 = training.p_0 * np.prod(np.where(x == 1, training.neg_probs, 1 - training.neg_probs), axis=1)
    p_1 = training.p_1 * np.prod(np.where(x == 1, training.pos_probs, 1 - training.pos_probs), axis=1)
    
    return p_1 > p_0

def evaluate_bayes(training_set, test_set, perc, no_graph=False):
    x_train, y_train = training_set
    x_test, y_test = test_set
    training_metrics = []
    test_metrics = []
    training_metrics_sklearn = []
    test_metrics_sklearn = []

    #for i in tqdm(range(perc, 101, perc)):
    for i in range(perc, 101, perc):
        set_sample = sample(list(range(len(x_train))), int(len(x_train) * i * 0.01))
        x_train_sample = np.array([x_train[i] for i in set_sample])
        y_train_sample = np.array([y_train[i] for i in set_sample])
        training = train_naive_bayes((x_train_sample, y_train_sample))
        if not no_graph:
            clf = BernoulliNB()
            clf.fit(x_train_sample, y_train_sample)

        training_metrics.append(calculate_metrics(x_train_sample, y_train_sample, training, test_naive_bayes))
        if not no_graph: training_metrics_sklearn.append(calculate_metrics(x_train_sample, y_train_sample, training, test_naive_bayes, clf))
        test_metrics.append(calculate_metrics(x_test, y_test, training, test_naive_bayes))
        if not no_graph: test_metrics_sklearn.append(calculate_metrics(x_test, y_test, training, test_naive_bayes, clf))
    
    if no_graph:
        return test_metrics[0][0]

    training_metrics = np.array(training_metrics)
    training_metrics_sklearn = np.array(training_metrics_sklearn)
    test_metrics = np.array(test_metrics)
    test_metrics_sklearn = np.array(test_metrics_sklearn)

    accuracy_graph(training_metrics[:, 0], test_metrics[:, 0], perc, 'Naive Bayes')
    accuracy_graph(training_metrics_sklearn[:, 0], test_metrics_sklearn[:, 0], perc, 'Naive Bayes')
    metrics_list = [training_metrics, training_metrics_sklearn, test_metrics, test_metrics_sklearn]
    titles = ['Training', 'Training (sklearn)', 'Test', 'Test (sklearn)']
    for i, metrics in enumerate(metrics_list):
        display_metrics(metrics, perc, titles[i], 'Naive Bayes')