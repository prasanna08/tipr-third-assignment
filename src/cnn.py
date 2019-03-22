import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score

class CNN(object):
    def __init__(self, filter_config, act_fn, input_shape, output_shape):
        self.filter_config = filter_config
        self.act_fn = act_fn
        self.act_fn_map = {
            'sigmoid': self.sigmoid,
            'swish': self.swish,
            'tanh': self.tanh,
            'relu': self.relu
        }
        self.sess = tf.Session()
        self.create_network(input_shape, output_shape)

    def swish_act_fn(self, inputs):
        beta = tf.Variable(tf.constant(0.1, shape=(1,)))
        inputs = inputs * tf.nn.sigmoid(beta * inputs)
        return inputs

    def relu_act_fn(self, inputs):
        return tf.nn.relu(inputs)

    def sigmoid_act_fn(self, inputs):
        return tf.nn.sigmoid(inputs)

    def tanh_act_fn(self, inputs):
        return tf.nn.tanh(inputs)

    def conv_layer(self, inputs, kernel_size, in_features, out_features, act_fn):
        w = tf.Variable(tf.truncated_normal(shape=(kernel_size, kernel_size, in_features, out_features), stddev=5e-2))
        b = tf.Variable(tf.constant(0.0, shape=(out_features,)))
        h = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding='SAME') + b
        h = self.act_fn_map[self.act_fn](h)
        h = tf.nn.max_pool(h, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        h = tf.nn.lrn(h, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        return h

    def flatten(self, inputs):
        return tf.reshape(inputs, [-1, tf.reduce_prod(inputs.shape[1:])])

    def fcn_layer(self, inputs, in_shape, out_shape):
        w = tf.Variable(tf.truncated_normal(shape=(in_shape, out_shape), stddev=0.04))
        b = tf.Variable(tf.constant(0.1, shape=(out_shape,)))
        return self.act_fn_map[self.act_fn](tf.matmul(inputs, w) + b)

    def softmax(self, logits):
        return tf.nn.softmax(logits, axis=1)

    def loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    def lr_schedular(self, lr, global_step):
        lr = tf.train.cosine_decay(0.001, global_step, 100000)
        return lr

    def optimizer_op(self, loss, lr, global_step):
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
        return optimizer

    def create_network(self, input_shape, output_shape):
        self.inputs = inputs = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(None, output_shape), dtype=tf.float32)
        x = self.inputs
        in_features = input_shape[-1]
        for kernel_size in filter_config:
            x = self.conv_layer(x, kernel_size, in_features, out_features=64)
            in_features = 64
            out_features = 64

        x = self.flatten(x)
        x = self.fcn_layer(x, x.shape[1], 384)
        x = self.fcn_layer(x, 384, 192)
        self.embedding_layer = x
        x = self.fcn_layer(x, 192, output_shape)
        x = self.softmax(x)
        self.output = x
        self.loss = self.loss(x, self.labels)

        self.global_step = tf.Variable(tf.constant(0), trainable=False)
        self.lr = self.lr_schedular(0.01, global_step)
        self.optimizer = self.optimizer_op(self, self.loss, self.lr, self.global_step)
        return self.optimizer

    def get_scores(self, predictions, labels):
        return {
            'f1-micro': f1_score(labels, predictions, average='micro'),
            'f1-macro': f1_score(labels, predictions, average='macro'),
            'accuracy': accuracy_score(labels, predictions)
        }

    def train(self, bg, max_epochs=10):
        epoch_cnt = 0
        steps = 0
        overall_loss = 0.0
        for (x, y, p) in bg:
            _, sloss = sess.run([self.optimizer, self.loss], feed_dict={self.inputs:x, self.labels:y})
            overall_loss += sloss
            steps += 1
            if p:
                output = sess.run([self.output], feed_dict={self.inputs:x})
                scores = self.get_scoores(output.argmax(axis=1), y.argmax(axis=1))
                epoch_cnt += 1
                print("Epoch %d, Loss: %.3f" % (epoch_cnt, overall_loss/steps))
        return

    def predict(self, inputs):
        return sess.run([self.output], feed_dict={self.inputs: inputs})[0]

    def evaluate(self, inputs, labels):
        predictions = self.predict(inputs)
        return self.get_scores(predictions.argmax(axis=1), labels.argmax(axis=1))
