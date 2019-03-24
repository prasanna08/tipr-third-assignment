import numpy as np

class BatchGenerator(object):
    def __init__(self, data, labels, batch_size, shape, split_ratio=(0.6, 0.8)):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.splitted = False
        self._cursor = 0
        self.split_ratio = split_ratio
        self.shape = shape
        self.split()
        
    def shuffle_data(self):
        idcs = np.arange(self.data.shape[0])
        np.random.shuffle(idcs)
        self.data = self.data[idcs]
        self.labels = self.labels[idcs]
    
    def split(self):
        self.shuffle_data()
        self.splitted = True
        instances = self.data.shape[0]
        i = int(self.split_ratio[0]*instances)
        j = int(self.split_ratio[1]*instances)
        self.train_data = self.data[:i]
        self.train_labels = self.labels[:i]
        self.valid_data = self.data[i:j]
        self.valid_labels = self.labels[i:j]
        self.test_data = self.data[j:]
        self.test_labels = self.labels[j:]
    
    def get_training_data(self):
        X = self.data if not self.splitted else self.train_data
        Y = self.labels if not self.splitted else self.train_labels
        return self.rescale(X.reshape(-1, *shape)), self.onehot(Y)

    def get_validation_data(self):
        if self.splitted:
            return self.rescale(self.valid_data.reshape(-1, *shape)), self.onehot(self.valid_labels)
        else:
            return None
    
    def get_test_data(self):
        if self.splitted:
            return self.rescale(self.test_data.reshape(-1, *shape)), self.onehot(self.test_labels)
        else:
            return None
        
    def rescale(self, x):
        return (x.astype(np.float32) - 128) / 255.0

    def onehot(self, y):
        labels = np.zeros(shape=(y.shape[0], 10))
        labels[np.arange(y.shape[0]), y[:, 0]] = 1
        return labels
    
    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self):
        X = self.data if not self.splitted else self.train_data
        Y = self.labels if not self.splitted else self.train_labels

        if self._cursor + self.batch_size > X.shape[0]:
            rem = self._cursor + self.batch_size - X.shape[0]
            batch_input = np.concatenate([X[self._cursor:], X[:rem]], axis=0)
            batch_output = np.concatenate([Y[self._cursor:], Y[:rem]], axis=0)
            self._cursor = rem
            return self.rescale(batch_input.reshape(self.batch_size, *shape)), self.onehot(batch_output), True

        batch_input = X[self._cursor: self._cursor+self.batch_size]
        batch_output = Y[self._cursor: self._cursor+self.batch_size]
        _pcursor = self._cursor
        self._cursor = (self._cursor + self.batch_size) % (X.shape[0])

        return self.rescale(batch_input.reshape(self.batch_size, *shape)), self.onehot(batch_output), True if _pcursor >= self._cursor else False
