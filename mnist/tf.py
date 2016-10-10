#import tensorflow as tf
# From https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/comments
# and https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
import numpy as np
import pandas as pd
import tensorflow as tf

def kaggle_output_format(data, predictions):
    output = []
    output.append(["ImageId","Label"])
    if len(data) != len(predictions):
        raise
    i = 0
    for row in data:
        #print(row, predictions[i])
        output.append([row, predictions[i]])
        i += 1
    with open('kaggle_output.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)
    return output

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class Stochastic():
    def __init__(self, batch_size, train, labels):
        self.batch_size = batch_size
        self.train = train
        self.labels = labels
        self.start = 0
        self.epochs = 0

    def next_batch(self):
        end = self.start + self.batch_size
        if end > len(self.train):
            self.next_epoch()
            end = self.start + self.batch_size
        print(self.start, end)
        x, y =  self.train[self.start: end], self.labels[self.start: end]
        self.start = end
        return x,y

    def next_epoch(self):
        self.epochs += 1
        print("epoch ", self.epochs)
        self.start = 0
        perm = np.arange(len(self.train))
        np.random.shuffle(perm)
        self.labels = self.labels[perm]
        self.train = self.train[perm]


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    images = train.iloc[:,1:].values
    images = images.astype(np.float)
    labels = train[[0]].values.ravel()
    labels = dense_to_one_hot(labels, 10)
    labels = labels.astype(np.uint8)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    y_ = tf.placeholder(tf.float32, shape=[None, 10]) # Should be 10?
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    stoch = Stochastic(100, images, labels)
    for i in range(1000):
        batch_xs, batch_ys = stoch.next_batch()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(correct_prediction)

