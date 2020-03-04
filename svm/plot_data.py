import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA



WINDOW_SIZE = 10

def load_data_from_csv(filepath):
    csv = pd.read_csv(filepath, usecols=["angle"], squeeze=True)

    train_data = []
    i = 0
    while i < len(data):
        if i+WINDOW_SIZE > len(data):
            break
        train_data.append(data[i:i+WINDOW_SIZE])
        i += WINDOW_SIZE

    return train_data


def make_meshgrid(x, y, h=.02):
    x_min, x_max = min(x) - 1, max(x) + 1
    y_min, y_max = min(y) - 1, max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

train_data = []
train_result = []

forward_fall = load_data_from_csv("./data/forward_fall.csv")
train_data.extend(forward_fall)
train_result.extend([1] * len(forward_fall))
side_fall = load_data_from_csv("./data/s_fall.csv")
train_data.extend(side_fall)
train_result.extend([1] * len(side_fall))
back_fall = load_data_from_csv("./data/backfall.csv")
train_data.extend(back_fall)
train_result.extend([1] * len(back_fall))

walking = load_data_from_csv("./data/walking.csv")
train_data.extend(walking)
train_result.extend([0] * len(walking))
bending = load_data_from_csv("./data/bending_and_standing.csv")
train_data.extend(bending)
train_result.extend([0] * len(bending))
sitting = load_data_from_csv("./data/sitting_down_standing_up.csv")
train_data.extend(sitting)
train_result.extend([0] * len(sitting))
jumping = load_data_from_csv("./data/jumping.csv")
train_data.extend(jumping)
train_result.extend([0] * len(jumping))
running = load_data_from_csv("./data/running.csv")
train_data.extend(running)
train_result.extend([0] * len(running))

# use pca to reduce the data dimension
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(train_data)

model = svm.SVC(kernel='linear')
clf = model.fit(reduced_data, train_result)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = reduced_data[:, 0], reduced_data[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=train_result, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
