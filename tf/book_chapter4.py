import input_data
import tensorflow as tf
import numpy as np
import cv2

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    y = tf.nn.softmax(tf.matmul(x, W)+b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # print sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


    # show weight img
    w = sess.run(W)
    img1 = w[:, 0]
    img1 = np.reshape(img1, (28, 28))
    cv2.imshow('img', img1)
    cv2.waitKey(0)
