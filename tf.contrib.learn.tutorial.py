# https://www.tensorflow.org/versions/r0.9/tutorials/tflearn/index.html
import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAIN = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# LOAD Data sets
training_set = tf.contrib.learn.datasets.base.load_csv(
    filename=IRIS_TRAIN,
    target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(
    filename=IRIS_TEST,
    target_dtype=np.int)

# X and Y is common in ML.
# X == features
# Y == targets (in this case categories)
x_train = training_set.data
x_test = test_set.data
y_train = training_set.target
y_test = test_set.target

# 3 layer DNN with 10, 20, 10 units respectively
classifier = tf.contrib.learn.DNNClassifier(
    hidden_units=[10, 20, 10],
    n_classes=3)

classifier.fit(x_train, y_train, steps=200)

accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print("Accuracy: {0:f}".format(accuracy_score))

# Maybe about 93% accuracy?

# Classify some new samples
new_samples = np.array(
    [
    [6.4, 3.2, 4.5, 1.5],
    [5.8, 3.1, 5.0, 1.7]], dtype=float)

y = classifier.predict(new_samples)
print("Predictions: {}".format(str(y)))
