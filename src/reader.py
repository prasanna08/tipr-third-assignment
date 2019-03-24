import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
import gzip

def read_pickle_file(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def read_cifar_dataset(data_root):
    x = []
    y = []
    for f in os.listdir(data_root):
        if f.endswith('meta') or f.endswith('html') or f.endswith('.DS_Store'):
            continue
        data = read_pickle_file(os.path.join(data_root, f))
        y.append(np.array(data[b'labels']).reshape(-1, 1))
        x.append(data[b'data'])
    dx = np.concatenate([i for i in x], axis=0).reshape(-1, 32, 32, 3)
    dy = one_hot_transform(np.concatenate([i for i in y], axis=0))
    return dx, dy

def one_hot_transform(labels):
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.unique(labels).reshape(-1, 1))
    encoded_labels = np.asarray(encoder.transform(np.array(labels).reshape(-1, 1)).todense()).astype(np.uint8)
    return encoded_labels

def read_fmnist_dataset(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28, 1)

    return images, one_hot_transform(labels)