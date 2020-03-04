import pandas as pd
from sklearn import svm


WINDOW_SIZE = 10


def load_data_from_csv(filepath):
    csv = pd.read_csv(filepath, usecols=["angle"], squeeze=True)
    data = list(csv.values)

    train_data = []
    i = 0
    while i < len(data):
        if i+WINDOW_SIZE > len(data):
            break
        train_data.append(data[i:i+WINDOW_SIZE])
        i += WINDOW_SIZE

    return train_data

train_data = []
train_result = []
forward_fall = load_data_from_csv("./data/forward_fall.csv")
side_fall = load_data_from_csv("./data/s_fall.csv")
walking = load_data_from_csv("./data/walking.csv")
jumping = load_data_from_csv("./data/jumping.csv")
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


bending = load_data_from_csv("./data/bending_and_standing.csv")
sitting = load_data_from_csv("./data/sitting_down_standing_up.csv")
back_fall = load_data_from_csv("./data/backfall.csv")
running = load_data_from_csv("./data/running.csv")
print(model.predict(bending))
print(model.predict(sitting))
print(model.predict(back_fall))

