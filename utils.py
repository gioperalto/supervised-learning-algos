import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def red_wine_quality():
    data = pd.read_csv('data/red-wine-quality/full.csv')
    columns = data.columns.drop('quality')
    x = pd.DataFrame(data, columns=columns)
    y = pd.DataFrame(data, columns=['quality'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    return x_train, y_train, x_test, y_test

def heart_failure_prediction():
    data = pd.read_csv('data/heart-failure-prediction/full.csv')
    columns = data.columns.drop('DEATH_EVENT')
    x = pd.DataFrame(data, columns=columns)
    y = pd.DataFrame(data, columns=['DEATH_EVENT'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    return x_train, y_train, x_test, y_test

def test(learner, x):
    return learner.predict(x)

def accuracy(learner, x, y):
    y_predictions = test(learner, x)
    
    return accuracy_score(y, y_predictions)*100

def get_accuracy(learners, x, y):
    accuracies = []

    for i in range(len(learners)):
        accuracies.append(accuracy(learners[i], x, y))

    return accuracies

def mse(learner, x, y):
    y_predictions = test(learner, x)

    return mean_squared_error(y, y_predictions)

def get_mse(learners, x, y):
    error_rates = []

    for i in range(len(learners)):
        error_rates.append(mse(learners[i], x, y))

    return error_rates

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    
    return scaler.transform(x_train), scaler.transform(x_test)

def benchmarks(x_train, y_train, x_test, y_test, overloads, test_mse, fn, count=1000):
    count, min_mse = count, min(test_mse)
    i = test_mse.index(min_mse)
    o = overloads[i]
    learner, times = fn(x_train, y_train, o), np.zeros((count))

    for x in range(count):
        create_start = time.process_time()
        learner = fn(x_train, y_train, o)
        create_time = time.process_time() - create_start
        times[x] = create_time

    print('Benchmark (Create): {}'.format((np.mean(times))*count))

    times = times * 0.

    for x in range(count):
        query_start = time.process_time()
        learner.predict(x_test)
        query_time = time.process_time() - query_start
        times[x] = query_time
    
    print('Benchmark (Query): {}'.format(np.mean(times)*count))
    print('Benchmark (Error): {}'.format(mse(learner, x_test, y_test)))