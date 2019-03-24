# op = sess.graph.get_tensor_by_name('CNN/softmax/Softmax:0')
# ip = sess.graph.get_tensor_by_name('CNN/placeholder/Placeholder:0')

import os
from sklearn.metrics import f1_score, accuracy_score
import argparse

import cnn
import batch_generators
import reader

def load_cnn_model(name):
	if name == 'fashion-mnist':
		model_root = os.path.join(cur_path, '../models/FMNIST/FMNIST-Best')
	elif name == 'cifar':
		model_root = os.path.join(cur_path, '../models/CIFAR/CIFAR-Best')
	sess = tf.Session()
	saver = tf.train.import_meta_graph('%s.meta' % model_root)
	saver.restore(sess, model_root)
	op = sess.graph.get_tensor_by_name('CNN/softmax/Softmax:0')
	ip = sess.graph.get_tensor_by_name('CNN/placeholder/Placeholder:0')
	return sess, ip, op

def predict(sess, input_images, ip, op):
	preds = sess.run([op], feed_dict={ip:input_images})[0]
	return preds

def score(predictions, labels):
	return {
        'f1-micro': f1_score(labels, predictions, average='micro'),
        'f1-macro': f1_score(labels, predictions, average='macro'),
        'accuracy': accuracy_score(labels, predictions)
    }

def print_scores(scores):
    print("Test Accuracy :: %.3f" % (scores['accuracy']))
    print("F1 Micro :: %.3f" % (scores['f1-micro']))
    print("F1 Macro :: %.3f" % (scores['f1-macro']))

if __name__ == '__main__':
    print('Welcome to the world of CNNs!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train-data', type=str)
    parser.add_argument('--filter-config', type=str)
	parser.add_argument('--activation', type=str)
    args = parser.parse_args()

    if args.train_data:
    	filter_config = [int(i) for i in args.filter_config[1:-1].split(' ')]
        if args.dataset.lower() == 'cifar':
    	   trainx, trainy = reader.read_cifar_dataset(args.train_data)
           testx, testy = reader.read_cifar_dataset(args.test_data)
        elif args.dataset.lower() == 'fashion-mnist':
            trainx, trainy = reader.read_fmnist_dataset(args.train_data)
            testx, testy = reader.read_fmnist_dataset(args.train_data, kind='t10k')
    	model = cnn.CNN(filter_config, args.activation.lower(), trainx.shape[1:], trainy.shape[1])
    	bg = batch_generators.BatchGenerator(trainx, trainy, 128, shape=trainx.shape[1:], split_ratio=(1.0, 0.0))
    	model.train(bg)
    	print_score(model.evaluate(testx, testy))
    else:
    	sess, ip, op = load_cnn_model(args.dataset.lower())
    	testx, testy = reader.read_dataset(args.test_data)
    	predy = predict(sess, testx, ip, op)
    	print_score(score(predy.argmax(axis=1), testy.argmax(axis=1)))
