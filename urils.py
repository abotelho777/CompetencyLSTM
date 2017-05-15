import pandas as pd
import numpy as np

def counter(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return unique, counts

def counter_dict(a):
    a = list(a)
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))

def get_user_index_mapping(dataList):
    a = list(dataList)
    target = [a[0], 0]
    size = len(a)
    for i in range(1, size):
        if a[i] != a[i - 1]:
            temp = [a[i], i]
            target = np.vstack((target, temp))
    print ('==> students number\t',len(target))
    return target


if __name__ == '__main__':
    k = [1,1,1,1,3,4,4,4,5,5,6,6]
    print (get_user_index_mapping(k))

