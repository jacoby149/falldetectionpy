import pandas as pd
import os

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
