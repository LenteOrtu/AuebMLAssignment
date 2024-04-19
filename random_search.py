from random import sample
from time import sleep
from multiprocessing import Process, Manager
from tqdm import tqdm
from utils import *
from MLP import rnn
import log_reg
import numpy as np
from sklearn.model_selection import train_test_split 
from naive_bayes import *
import itertools

# Hyperparameters:
# n, m, k
# Logistic regression: learning_rate, lambda
n = k = 15000
learning_rate = 0.01
def generate_random_numbers(n, low, upper, step):
    nums = [round(val, 4) for val in list(np.arange(low, upper + step, step))]
    return sorted(sample(nums, n))

def get_combinations(n_m, n_lambda):
    random_m = generate_random_numbers(n_m, 1000, 20000, 1000)

    random_lambda = generate_random_numbers(n_lambda, 0.001, 0.01, 0.001)
    combinations = list(itertools.product(random_m, random_lambda))

    return combinations

def process_search(combinations, return_list):
    min_mean_error = 1
    best_combination = None
    val_size = 0.2

    for combination in tqdm(combinations):
        m, l = combination
        train, _ = preprocess_data(n, m, k)

        x_train, x_test , y_train, y_test = train_test_split(train[1], train[2], test_size=val_size)
        bayes_error = evaluate_bayes((x_train, y_train), (x_test, y_test), 100, True)
        log_reg_error = log_reg.evaluate_logistic_regression((x_train, y_train), (x_test, y_test), 100, 100, learning_rate, l, True)
        x_train, x_test, y_train, y_test = train_test_split(train[0], train[2], test_size=val_size)
        rnn_error = rnn((x_train, y_train), (x_test, y_test), m, 100, 10, True)

        curr_min_error = (bayes_error + log_reg_error + rnn_error) / 3
        if curr_min_error < min_mean_error:
            min_mean_error = curr_min_error
            best_combination = (n, k, m, learning_rate, l)
    
    return_list.append((1 - min_mean_error, best_combination))
    #return 1 - min_mean_error, best_combination
    
def search(num_of_processes):
    combinations = get_combinations(2, 3)
    manager = Manager()
    return_list = manager.list()

    processes = []
    step = len(combinations) // num_of_processes
    for i in range(num_of_processes):
        if i != num_of_processes - 1: curr_combinations = combinations[i*step : (i+1)*step]
        else: curr_combinations = combinations[i*step:]
        p = Process(target=process_search, args=(curr_combinations, return_list))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    
    max_idx = return_list.index(max(return_list, key=lambda x: x[0]))
    print(process_search(combinations, []))
    print(return_list[max_idx])

if __name__ == '__main__':
    search(6)