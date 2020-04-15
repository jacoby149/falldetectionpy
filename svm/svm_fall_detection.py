import pandas as pd
from sklearn import svm
import math, os, time
from frequency_domain import *
import sys

# for 10hz sampling rate
WINDOW_SIZE = 10 # 0.1s interval,s 1s sliding window
TRAIN_DATA_DIR = "./data/"


def calculate_magnitude(csv):
    num_rows = len(csv.index)
    magnitudes = []
    for i in range(num_rows):
        (x, y, z) = (csv.x[i], csv.y[i], csv.z[i])
        magnitudes.append((x**2 + y**2 + z**2) ** 0.5)
    return magnitudes[1:]


def calculate_angle(csv):
    num_rows = len(csv.index)
    z_angles = []
    prev_z = 0
    for i in range(num_rows):
        (x, y, z) = (csv.x[i], csv.y[i], csv.z[i])
        z_angle = math.atan(z / ((x**2 + y**2) ** 0.5))
        if i == 0:
            prev_z = z_angle
        else:
            z_angles.append(abs(z_angle - prev_z))
    return z_angles


def load_data_from_csv(filepath):
    columns = ["x", "y", "z", "is_fall"]
    csv = pd.read_csv(filepath, usecols=columns, squeeze=True)
    num_rows = len(csv.index)
    magnitudes = calculate_magnitude(csv)
    z_ang = calculate_angle(csv)

    # break up data into windows
    train_X = []
    train_Y = []
    for i in range(0, len(magnitudes)-WINDOW_SIZE+1):
        window = magnitudes[i:i+WINDOW_SIZE] + z_ang[i:i+WINDOW_SIZE]
        train_X.append(window)

        # check if there is a fall within the window
        is_fall = 0
        for j in range(i, i+WINDOW_SIZE):
            if csv.is_fall[j]:
                is_fall = 1
                break
        train_Y.append(is_fall)
    return train_X, train_Y


class SVM():

    def __init__(self):
        train_X = []
        train_Y = []
        for filename in os.listdir(TRAIN_DATA_DIR):
            if filename.endswith(".csv"):
                filepath = os.path.join(TRAIN_DATA_DIR, filename)
                x, y = load_data_from_csv(filepath)
                train_X.extend(x)
                train_Y.extend(y)
        self.model = svm.SVC()
        self.model.fit(train_X, train_Y)

    def predict(self, csv):
        z_ang = calculate_angle(csv)
        magnitudes = calculate_magnitude(csv)

        if len(magnitudes) < WINDOW_SIZE:
            return

        windows = []
        for i in range(0, len(magnitudes)-WINDOW_SIZE+1):
            window = magnitudes[i:i+WINDOW_SIZE] + z_ang[i:i+WINDOW_SIZE] #+
            windows.append(window)

        prediction = self.model.predict(windows)
        return prediction


if __name__ == '__main__':
    model = SVM()
    testfile = './testdata/mixed.csv'
    csv = pd.read_csv(testfile, squeeze=True)
    prediction = model.predict(csv)
    print(prediction)


#mixed: fall, fall, walking, jumping

