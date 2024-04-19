import tensorflow as tf
from utils import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tqdm import tqdm
from random import sample
import numpy as np
from keras import backend as K
from graph import *

def recall_m(y, y_hat):
    TP = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y, 0, 1)))
    return TP / (all_positives + K.epsilon())

def precision_m(y, y_hat):
    TP = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
    predicted_pos = K.sum(K.round(K.clip(y_hat, 0, 1)))
    return TP / (predicted_pos + K.epsilon())

def rnn(train_set, test_set, max_vocab, perc, epochs, no_graph=False):
    x_train, y_train = train_set
    x_test, y_test = test_set
    training_metrics = []
    test_metrics = []
    history = None

    #for i in tqdm(range(perc, 101, perc)):
    for i in range(perc, 101, perc):
        set_sample = list(sample(list(range(len(x_train))), int(len(x_train) * 0.01 * i)))
        x_sample = np.array([x_train[i] for i in set_sample])
        y_sample = np.array([y_train[i] for i in set_sample])

        model, history = create_model((x_sample, y_sample), max_vocab, epochs)
        loss, accuracy, precision, recall = model.evaluate(x_sample, y_sample)
        training_metrics.append((1 - accuracy, precision, recall))

        loss, accuracy, precision, recall = model.evaluate(x_test, y_test)
        test_metrics.append((1 - accuracy, precision, recall))
    
    if no_graph:
        return test_metrics[0][0]

    training_metrics = np.array(training_metrics)
    test_metrics = np.array(test_metrics)

    accuracy_graph(training_metrics[:, 0], test_metrics[:, 0], perc, 'RNN')
    display_metrics(training_metrics, perc, 'Training', 'RNN')
    display_metrics(test_metrics, perc, 'Test', 'RNN')
    plot_epochs(history, 'RNN')

def get_average_length(x_train):
    avg_len = 0
    for example in x_train: avg_len += len(str(example).split())
    return int(avg_len / len(x_train))

def create_model(train, max_vocab, epochs):
    x_train, y_train = train
    avg_length = get_average_length(x_train)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_vocab, output_sequence_length=avg_length)
    vectorizer.adapt(x_train)

    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(
            input_dim=len(vectorizer.get_vocabulary()),
            output_dim=64,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy', precision_m, recall_m])
    
    history = model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1, batch_size=100, validation_split=0.2)

    return model, history
