import tensorflow as tf
import os


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
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked = read_csv(100, "data/train.csv",
                                                [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0]])
    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ["female"]))
    # Finally we pack all the features in a single matrix;
    # We then transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.pack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    return features, survived


def inference(w, x, b):
    return tf.matmul(x, w) + b


def loss(w, x, b, y):
    return tf.nn.softmax_cross_entropy_with_logits(inference(w, x, b), y)


def train(lr, cost):
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    return train_op


def evluation(w, x, b, sess, Y):
    predictive = tf.cast(inference(w, x, b)>0.5, tf.float32)
    print sess.run(tf.reduce_mean(tf.cast(tf.equal(predictive, Y), tf.float32)))

if __name__ == '__main__':
    pass