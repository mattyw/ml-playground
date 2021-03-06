#  Estimate Pi
# Inspired by https://github.com/jostmey/NakedTensor/blob/master/serial.py
#
# Given some cirumferences and some diameters we want to see if we can "learn" the value of pi

import tensorflow as tf
import numpy as np
import random
import math

def jitter(ls):
    choice = [1, -1, 0.1, -0.1, 0.01, -0.01, 0.001, -0.001]
    return [x+random.choice(choice) for x in ls]


#cs = [157.08, 150.796, 128.805, 109.956, 65.973, 116.239, 147.655, 147.655, 131.947, 128.805]  # A list of circumferences
#cs = [157.1, 150.8, 128.8, 110.0, 66.0, 116.2, 147.7, 147.7, 131.9, 128.8]

#cs = [157.0, 150.0, 128.0, 110.0, 66.0, 116.0, 147.0, 147.0, 131.0, 128.0]
#ds = [50.0, 48.0, 41.0, 35.0, 21.0, 37.0, 47.0, 47.0, 42.0, 41.0]  # A list of diameters for the given cirumference
cs = jitter([random.randint(20, 100) for x in range(1000)])
ds = jitter([round(c/math.pi, 10) for c in cs])

estimates = [item[0]/item[1] for item in zip(cs,ds)]

print("Estimates from equation:", estimates)  # If we just esitmate based on pi = c/d we get these estimates for pi

print("Mean pi estimate: ", np.mean(estimates))


pi_initial = 2.123  # Initial guess


p = tf.Variable(pi_initial)  # Free variable to be solved

error = 0.0
for i in range(len(cs)):
    d_model = cs[i] / p  # The model d=c/pi
    error += (ds[i] - d_model)**2  # Difference squared - this is the "cost" to be minimized


'''
once cost function is defined, use gradient descent to find global minimum.
'''

operation = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(error)

with tf.Session() as session:
    session.run(tf.initialize_all_variables())  # Initialize session

    epochs = 10000  # number of "sweeps" across data

    for iteration in range(epochs):
        session.run(operation)
    print("Learned Pi:", p.eval())
