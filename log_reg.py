from utils import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from random import sample
from graph import *
import pandas as pd
from tqdm import tqdm

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(x_train, y_train, batch_size, epochs, learning_rate, lambda_param):
    m, n = x_train.shape

    w = np.zeros((n, 1))
    y_train = y_train.reshape(m, 1)

    for epoch in range(epochs):
        for i in range((m - 1) // batch_size + 1):
            start = i * batch_size
            x_train_batch = x_train[start:start + batch_size]
            y_train_batch = y_train[start:start + batch_size]

            p = sigmoid(np.dot(x_train_batch, w))
            w += learning_rate * np.dot(x_train_batch.T, (y_train_batch - p)) - 2 * learning_rate * lambda_param * w

    return w

def predict(x, w):
    preds = np.dot(x, w)
    pred_class = [1 if i >= 0 else 0 for i in preds]

    return np.array(pred_class)

def evaluate_logistic_regression(training_set, test_set, percentage_increase, max_iter, learning_rate=0.01, lambda_=0.001, no_graph=False):
    x_train, y_train = training_set
    x_test, y_test = test_set
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)
    
    training_metrics = []
    training_metrics_sklearn = []
    test_metrics = []
    test_metrics_sklearn = []

    #for i in tqdm(range(percentage_increase, 101, percentage_increase)):
    for i in range(percentage_increase, 101, percentage_increase):
        set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * 0.01 * i)))
        x_train_sample = np.array([x_train[i] for i in set_sample])
        y_train_sample = np.array([y_train[i] for i in set_sample])
        w = train(x_train_sample, y_train_sample, batch_size=100, epochs=max_iter, learning_rate=learning_rate, lambda_param=lambda_)
        if not no_graph:
            clf = LogisticRegression(max_iter=max_iter)
            clf.fit(x_train_sample, y_train_sample)

        training_metrics.append(calculate_metrics(x_train_sample, y_train_sample, w, predict))
        if not no_graph: training_metrics_sklearn.append(calculate_metrics(x_train_sample, y_train_sample, w, predict, clf))
        test_metrics.append(calculate_metrics(x_test, y_test, w, predict))
        if not no_graph: test_metrics_sklearn.append(calculate_metrics(x_test, y_test, w, predict, clf))

    if no_graph:
        return test_metrics[0][0]

    training_metrics = np.array(training_metrics)
    training_metrics_sklearn = np.array(training_metrics_sklearn)
    test_metrics = np.array(test_metrics)
    test_metrics_sklearn = np.array(test_metrics_sklearn)

    accuracy_graph(training_metrics[:, 0], test_metrics[:, 0], percentage_increase, 'Logistic regression')
    accuracy_graph(training_metrics_sklearn[:, 0], test_metrics_sklearn[:, 0], percentage_increase, 'Logistic regression (sklearn)')
    metrics_list = [training_metrics, training_metrics_sklearn, test_metrics, test_metrics_sklearn]
    titles = ['Training', 'Training (sklearn)', 'Test', 'Test (sklearn)']
    for i, metrics in enumerate(metrics_list):
        display_metrics(metrics, percentage_increase, titles[i], 'Logistic regression')