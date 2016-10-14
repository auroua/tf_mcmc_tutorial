import tensorflow as tf
import os


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'learning rate')


class Iris(object):
    def __init__(self):
        self.w = tf.Variable(tf.zeros[4, 3], name='weights')
        self.b = tf.Variable(tf.zeros[3], name='bias')

    def inference(self, features):
        return tf.matmul(features, self.w) + self.b

    def loss(self, logit, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels)

    def train(self, cost):
        return tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    def evaluate(self, features, labels):
        predictive = tf.cast(tf.argmax(tf.inference(features), 1), tf.int32)
        return tf.reduce_mean(tf.cast(tf.equal(predictive, labels), tf.float32))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)
    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label = read_csv(100, "data/iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
    # convert class names to a 0 based class index.
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.pack([tf.equal(label, ["Iris-setosa"]),
    tf.equal(label, ["Iris-versicolor"]), tf.equal(label, ["Iris-virginica"])])), 0))
    # Pack all the features that we care about in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.pack([sepal_length, sepal_width, petal_length, petal_width]))
    return features, label_number


if __name__ == '__main__':
    features, labels = inputs()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    np_features = sess.run(features)
    # np_labels = sess.run(labels)
    print np_features.shape
    # print np_labels.shape

