import numpy as np 
import matplotlib.pyplot as plt 

def compute_lag_metric(actual, prediction, lookup, symbol):
    diff_list = [None] * lookup
    lag_list = [None] * (len(actual)-lookup+1)

    for i in range(len(actual)-lookup+1):
        for j in range(lookup):
            diff_list[j] = abs(actual[i] - prediction[i+j])
        lag_list[i] = diff_list.index(min(diff_list))

    max_diff_count = [0] * lookup

    for i in range(len(lag_list)):
        max_diff_count[lag_list[i]] += 1

    _, ax = plt.subplots()
    ax.bar(range(len(max_diff_count)), max_diff_count, align='center')
    plt.sca(ax)
    plt.title(symbol+" Lag Test")
    ax.set_xlabel('Day Lag')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    _, ax1 = plt.subplots()
    ax1.scatter(range(len(lag_list)), lag_list)
    plt.title(symbol+" Daily Lag Test")
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Lag')
    ax1.grid(True)

    return lag_list