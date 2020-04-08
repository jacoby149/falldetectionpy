import pandas as pd
from sklearn import svm
import math, os, time
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import animation as animation
from frequency_domain import *
import sys

# for 10hz sampling rate
WINDOW_SIZE = 10 # 0.1s interval, so 1s sliding window
TRAIN_DATA_DIR = "./data/"


def calculate_magnitude(csv):
    num_rows = len(csv.index)
    magnitudes = []
    for i in range(num_rows):
        (x, y, z) = (csv.x[i], csv.y[i], csv.z[i])
        magnitudes.append((x**2 + y**2 + z**2) ** 0.5)
    return magnitudes

def calculate_angle(csv):
    num_rows = len(csv.index)
    x_angles = []
    y_angles = []
    z_angles = []
    prev_x, prev_y, prev_z = 0, 0, 0
    for i in range(num_rows):
        (x, y, z) = (csv.x[i], csv.y[i], csv.z[i])
        x_angle = math.atan(x / ((y**2 + z**2) ** 0.5))
        y_angle = math.atan(y / ((x**2 + z**2) ** 0.5))
        z_angle = math.atan(z / ((x**2 + y**2) ** 0.5))
        if i == 0:
            prev_x, prev_y, prev_z = x_angle, y_angle, z_angle
        else:
            x_angles.append(abs(x_angle - prev_x))
            y_angles.append(abs(y_angle - prev_y))
            z_angles.append(abs(z_angle - prev_z))
    return x_angles, y_angles, z_angles

def load_data_from_csv(filepath):
    columns = ["x", "y", "z", "is_fall"]
    csv = pd.read_csv(filepath, usecols=columns, squeeze=True)
    num_rows = len(csv.index)
    magnitudes = calculate_magnitude(csv)
    x_ang, y_ang, z_ang = calculate_angle(csv)

    # break up data into windows
    train_X = []
    train_Y = []
    for i in range(0, len(magnitudes)-WINDOW_SIZE):
        # max_mag = max(magnitudes[i:i+WINDOW_SIZE])
        # max_ang = max(max(x_ang[i:i+WINDOW_SIZE]), max(y_ang[i:i+WINDOW_SIZE]),
        #               max(z_ang[i:i+WINDOW_SIZE]))
        window = (magnitudes[i:i+WINDOW_SIZE] + z_ang[i:i+WINDOW_SIZE])# +
                  #x_ang[i:i+WINDOW_SIZE] + y_ang[i:i+WINDOW_SIZE])
        #+ add_freq_features(magnitudes[i:i+WINDOW_SIZE])
        train_X.append(window)
        # check if there is a fall within the window
        is_fall = 0
        for j in range(i, i+WINDOW_SIZE):
            if csv.is_fall[j]:
                is_fall = 1
                break
        train_Y.append(is_fall)
    return train_X, train_Y


def predict(magnitude, x_ang, y_ang, z_ang):
    train_X = []
    train_Y = []
    for filename in os.listdir(TRAIN_DATA_DIR):
        if filename.endswith(".csv"):
            filepath = os.path.join(TRAIN_DATA_DIR, filename)
            x, y = load_data_from_csv(filepath)
            train_X.extend(x)
            train_Y.extend(y)
    model = svm.SVC()
    model.fit(train_X, train_Y)
    windows = []
    for i in range(0, len(magnitudes)-WINDOW_SIZE):
        # max_mag = max(magnitudes[i:i+WINDOW_SIZE])
        # max_ang = max(max(x_ang[i:i+WINDOW_SIZE]), max(y_ang[i:i+WINDOW_SIZE]),
        #               max(z_ang[i:i+WINDOW_SIZE]))
        window = (magnitudes[i:i+WINDOW_SIZE] + z_ang[i:i+WINDOW_SIZE])# +
                 # x_ang[i:i+WINDOW_SIZE] + y_ang[i:i+WINDOW_SIZE])
        #+ add_freq_features(magnitudes[i:i+WINDOW_SIZE])
        # windows.append((max_mag, max_ang))
        windows.append(window)
    prediction = model.predict(windows)
    return prediction


def visualize(magnitudes, prediction):

    # plot time vs magnitude
    intervals = [0.1 * x for x in range(len(magnitudes))]
    fig = plt.figure()
    plt.plot(intervals, magnitudes)
    plt.xlabel("time (s)")
    plt.ylabel("acceleration")
    ax = plt.gca()

    top = max(magnitudes) # top position
    bottom = min(magnitudes)
    width = len(magnitudes) // WINDOW_SIZE

    rectangles = []
    # break up data into sliding windows and add rectangles for each sliding window
    for i in range(0, len(magnitudes)-WINDOW_SIZE):
        rectangles.append(patches.Rectangle((i/WINDOW_SIZE, bottom), 1, top-bottom,
                          linewidth=1, edgecolor='r', facecolor='none', zorder=3))

    # store texts for each prediction
    texts = []
    for p in prediction:
        if p == 1:
            texts.append("fall")
        else:
            texts.append("normal")

    # draw first rectangle and prediction
    ax.add_patch(rectangles[0])
    ax.set_title(texts[0])

    running = True
    # updates the slider
    def update(i):
        if len(rectangles) > 1:
            rectangles[0].remove()
            rectangles.pop(0)
            ax.add_patch(rectangles[0])
            texts.pop(0)
            ax.set_title(texts[0])

    # starts the animation on mouse click
    def onClick(event):
        nonlocal running
        if running:
            ani.event_source.stop()
            running = False
        else:
            ani.event_source.start()
            running = True

    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = animation.FuncAnimation(fig, update, interval=50)
    plt.show()



if __name__ == '__main__':
    testfile = sys.argv[1]
    csv = pd.read_csv(testfile, squeeze=True)
    x_ang, y_ang, z_ang = calculate_angle(csv)
    magnitudes = calculate_magnitude(csv)
    prediction = predict(magnitudes, x_ang, y_ang, z_ang)
    visualize(magnitudes, prediction)

#mixed: fall, fall, walking, jumping

