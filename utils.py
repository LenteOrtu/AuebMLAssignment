import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from graph import *
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

def preprocess_data(n, m, k):
    index_from = n
    seed = 113
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(index_from=index_from, seed=seed)
    word_index = tf.keras.datasets.imdb.get_word_index()
    index2word = dict((i+3, word) for (word, i) in word_index.items())
    index2word[0] = '[pad]'
    index2word[1] = '[bos]'
    index2word[2] = '[oov]'
    max_idx = max(index2word)
    idxes = list(range(max_idx, max_idx-k, -1))
    for idx in idxes: del index2word[idx]

    x_train = np.array([' '.join([index2word[idx] if idx in index2word else '[oov]' for idx in text]) for text in x_train])
    selector = SelectPercentile(mutual_info_classif, percentile=30)
    x_test = np.array([' '.join([index2word[idx] if idx in index2word else '[oov]' for idx in text]) for text in x_test])
    binary_vectorizer = CountVectorizer(binary=True, max_features=m)
    x_train_binary = binary_vectorizer.fit_transform(x_train).toarray()
    x_test_binary = binary_vectorizer.transform(x_test).toarray()
    selector.fit(x_train_binary, y_train)
    x_train_binary_fs = selector.transform(x_train_binary)
    x_test_binary_fs = selector.transform(x_test_binary)

    return (x_train, x_train_binary_fs, y_train), (x_test, x_test_binary_fs, y_test)

def calculate_metrics(x, y, training, predict_func, clf=None):
    prediction = predict_func(x, training) if clf is None else clf.predict(x)
    error = 1 - (np.sum(y == prediction) / len(y))
    TP = np.sum(np.logical_and(y == 1, prediction == 1))
    FP = np.sum(np.logical_and(y == 0, prediction == 1))
    FN = np.sum(np.logical_and(y == 1, prediction == 0))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return error, precision, recall

def display_metrics(metrics, perc, title, model_title):
    F_1 = (2 * metrics[:, 1] * metrics[:, 2]) / (metrics[:, 1] + metrics[:, 2])
    table = np.array([[1-v for v in metrics[:, 0]], metrics[:, 1], metrics[:, 2], F_1])
    cols = [str(a)+'%' for a in range(perc, 101, perc)]
    idx = ['Accuracy', 'Precision', 'Recall', 'F1']
    df = pd.DataFrame(table, columns=cols, index=idx)

    print(f'\n\n=== Metrics for {title} ===')
    print(df)
    prec_rec_graph(metrics[:, 1], metrics[:, 2], perc, title, model_title)