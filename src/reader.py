import os
import skimage
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def read_dataset(dpath):
    x = []
    y = []
    for i in os.listdir(dpath):
        print('reading images from %s folder.' % (i))
        for image in os.listdir(os.path.join(dpath, i)):
            if image.endswith('.jpg'):
                img = skimage.io.imread(os.path.join(dpath, i, image), as_gray=True)
                if resize:
                    img = skimage.transform.resize(img, resize)
                x.append(img)
                y.append(i)

    data_x = np.array(x)
    data_y = one_hot_transform(y)
    return data_x, data_y

def one_hot_transform(labels):
    encoder = OneHotEncoder(categories='auto')
    encoder.fit(np.unique(labels).reshape(-1, 1))
    encoded_labels = np.asarray(encoder.transform(np.array(labels).reshape(-1, 1)).todense()).astype(np.uint8)
    return encoded_labels
