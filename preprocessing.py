import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from sklearn.cluster import MiniBatchKMeans
from zipfile import BadZipFile

def prepare_data():
    """
    Get mnist data and preprocess
    :return: x_train np.array [60000 , 28, 28 , 1]
    :return: x_test np.array  [10000 , 28, 28 , 1]
    :return: y_train np.array [60000 , ]
    :return: y_test np.array  [10000 , ]
    """

    # Sometimes there is a stupid bug in keras loader we need to retry
    flag = True
    while flag:
        try:
            (x_train, y_train), (x_test, y_test) = load_data()
            flag = False
        except BadZipFile:
            pass
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test, y_train, y_test


def k_means(x):
    """
    Fit K-meanas clustering on image
    :param x: np.array(samples, 28,28,1)
    :return:
    """
    x = x.reshape(len(x), -1)  # reshape (samples, 28*28)
    kmeans = MiniBatchKMeans(n_clusters=10, batch_size=100, max_iter=1000, random_state=1)
    kmeans.fit(x)
    return kmeans


def check_kmeans(y, km_labels):
    """
    Check clusters of K means
    :param y: np.array (samples, ) of true labels
    :param km_labels: np.array (samples, ) of assigned km cluster
    :return:
    """
    unique_labels = np.unique(km_labels)
    for label in unique_labels:
        # Get y indices that match the current label
        y_ = y[np.where(km_labels == label)]
        # Count % of true labels in each cluster.
        # A good clustering means one true label of each km label
        hist = np.bincount(y_) / np.bincount(y_).sum(axis=0)
        print(np.round(hist, 2), np.argmax(hist))


def assign_to_clusters(x, y):
    """
    Assigns each image to it's arm (defined by KM clustering)
    :param x: np.array (samples , 28 ,28 ,1)
    :param y: np.array (samples , 28 ,28 ,1)

    :return: dict {key : index of arm , value : {x: (samples, 28,28,1), y: (samples,)}
    """

    n_samples = x.shape[0]
    km = k_means(x)
    arms = {arm: {'x': [], 'y': []} for arm in range(km.n_clusters)}
    for i in range(n_samples):
        # add x and y to each arm
        assigned_arm = km.labels_[i]
        arms[assigned_arm]['x'].append(x[i, :, :, :])
        arms[assigned_arm]['y'].append(y[i])
    # stack arrays x and y for each arm
    for arm in arms.keys():
        arms[arm]['x'] = np.stack(arms[arm]['x'], axis=0)
        arms[arm]['y'] = np.array(arms[arm]['y'])

    return arms
