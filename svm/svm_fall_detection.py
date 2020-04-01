import pandas as pd
from sklearn import svm
from matplotlib import pyplot as plt
from load_data import *

train_data = []
train_result = []
#five_fall = load_data_from_csv("./data/five_fall.csv")
col = "magnitude"
forward_fall = load_data_from_csv("./data/forward_fall.csv",col)
side_fall = load_data_from_csv("./data/s_fall.csv",col)
walking = load_data_from_csv("./data/walking.csv",col)
jumping = load_data_from_csv("./data/jumping.csv",col)
train_data.extend(forward_fall)
train_result.extend([1] * len(forward_fall))
train_data.extend(side_fall)
train_result.extend([1] * len(side_fall))
train_data.extend(walking)
train_result.extend([0] * len(walking))
train_data.extend(jumping)
train_result.extend([0] * len(jumping))

model = svm.SVC()
model.fit(train_data, train_result)


bending = load_data_from_csv("./data/bending_and_standing.csv",col)
sitting = load_data_from_csv("./data/sitting_down_standing_up.csv",col)
back_fall = load_data_from_csv("./data/backfall.csv",col)
running = load_data_from_csv("./data/running.csv",col)
print(model.predict(bending))
print(model.predict(sitting))
print(model.predict(back_fall))

