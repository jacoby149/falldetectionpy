import pandas as pd
import os
import numpy as np

WINDOW_SIZE = 10
SLIDE_INTERVAL = 2


class Data():

    def __init__(self, filepath, window_size):            
        csv = pd.read_csv(filepath, usecols=["vector"], squeeze=True)
        data = list(csv.values)

        train_data = []
        i = 0
        while i < len(data):
            if i+window_size > len(data):
                break
            train_data.append(data[i:i+window_size])
            i += window_size

        self.train_data = train_data
        
def load_data_from_csv(filepath,cols):
    csv = pd.read_csv(filepath, usecols=[cols], squeeze=True)
    data = list(csv.values)
    #plt.plot(data)
    #plt.show()

    train_data = []
    i = 0
    while i < len(data):
        if i+WINDOW_SIZE > len(data):
            break
        train_data.append(data[i:i+WINDOW_SIZE])
        i += SLIDE_INTERVAL

    return np.asarray(train_data)

