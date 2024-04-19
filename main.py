import numpy as np
from naive_bayes import *
import log_reg
from sklearn.model_selection import train_test_split 
from MLP import rnn
from utils import preprocess_data

# Hyperparameters are set based on results from random_search
def main(n, m, k):
    (x_train, x_train_binary, y_train), (x_test, x_test_binary, y_test) = preprocess_data(n, m, k)
    percentage_increase = 10 

    print('=' * 15 + ' Naive Bayes ' + '=' * 15)
    evaluate_bayes((x_train_binary, y_train), (x_test_binary, y_test), percentage_increase)
    print('=' * 15 + ' Logistic Regression ' + '=' * 15)
    log_reg.evaluate_logistic_regression((x_train_binary, y_train), (x_test_binary, y_test), percentage_increase, 100, 0.01, 0.001)
    print('=' * 15 + ' RNN (with LSTM) ' + '=' * 15)
    rnn((x_train, y_train), (x_test, y_test), m, percentage_increase, 10)

main(15000, 6000, 15000)