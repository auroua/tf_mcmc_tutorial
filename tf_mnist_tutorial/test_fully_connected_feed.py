import tensorflow as tf
import input_data
import mnist
import math
import time
import os

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('max_iter', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2')
flags.DEFINE_integer('batch_size', 100, 'batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')

def place_holder_inputs(batch_size):
    image_placeholder = tf.placeholder(tf.float32, shape=[batch_size, mnist.IMAGE_PIXELS])
    label_placeholder = tf.placeholder(tf.float32, shape=[batch_size])
    return image_placeholder, label_placeholder

def fill_feed_dict(data_set, image_pl, label_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
    feed_dict = {image_pl:images_feed, label_pl:labels_feed}
    return feed_dict

def inference(images):
    # hidden1
    with tf.name_scope('hidden1'):
        weights1 = tf.Variable(tf.truncated_normal([mnist.IMAGE_PIXELS, FLAGS.hidden1],
                                                    stddev=1.0 / math.sqrt(float(mnist.IMAGE_PIXELS)), name='weights'))
        bias1 = tf.Variable(tf.zeros([FLAGS.hidden1]), name='bias')
        hidden1_output = tf.nn.relu(tf.matmul(images, weights1) + bias1)

    #hidden2
    with tf.name_scope('hidden2'):
        weights2 = tf.Variable(tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2], stddev=1.0/math.sqrt(float(FLAGS.hidden1))), name='weights')
        bias2 = tf.Variable(tf.zeros(FLAGS.hidden2), name='bias')
        hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, weights2) + bias2)

    #softmax_layer
    with tf.name_scope('softmax-layer'):
        weights3 = tf.Variable(tf.truncated_normal([FLAGS.hidden2, 10], stddev=1.0/math.sqrt(float(FLAGS.hidden2))), name='weights')
        bias3 = tf.Variable(tf.zeros([10]), name='bias')
        logit = tf.matmul(hidden2_output, weights3) + bias3
    return logit

def loss(logit, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, labels, name='xenctropy')
    cost = tf.reduce_mean(cross_entropy, name='entropy_mean')
    return cross_entropy, cost

def train(cost, learning_rate):
    tf.scalar_summary(cost.op.name, cost)
    train = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = train.minimize(cost, global_step)
    return train_op


def evaluation(logist, labels):
    labels = tf.to_int32(labels)
    correct = tf.nn.in_top_k(logist, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / (1.0*num_examples)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        image_placeholder, bias_placeholder = place_holder_inputs(FLAGS.batch_size)
        logit = inference(image_placeholder)
        cross_entropy, cost = loss(logit, bias_placeholder)
        train_op = train(cost, FLAGS.learning_rate)
        eval_correct = evaluation(logit, bias_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        sess = tf.Session()
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        for step in range(FLAGS.max_iter):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       image_placeholder,
                                       bias_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, cost],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.

                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        image_placeholder,
                        bias_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        image_placeholder,
                        bias_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        image_placeholder,
                        bias_placeholder,
                        data_sets.test)


def main(_):
  run_training()


if __name__ == '__main__':
    tf.app.run()