import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('variable'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name='total_output')

        with tf.name_scope('transformation'):
            with tf.name_scope('input_layer'):
                a = tf.placeholder(tf.float32, shape=[None], name='input_a')
            with tf.name_scope('intermediate_layer'):
                b = tf.reduce_prod(a, name='pred_b')
                c = tf.reduce_sum(a, name='sum_c')
            with tf.name_scope('output'):
                output = tf.add(b, c, name='output')

        with tf.name_scope('update'):
            update_total = total_output.assign_add(output)
            increment_step = global_step.assign_add(1)

        with tf.name_scope('summaries'):
            avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name='average')
            tf.scalar_summary('Output', output, name='output_summary')
            tf.scalar_summary('Sum of outputs over time', update_total, name='total_summary')
            tf.scalar_summary('Average of outputs over time', avg, name='average_summary')

        with tf.name_scope('global_ops'):
            init = tf.initialize_all_variables()
            merged_summaries = tf.merge_all_summaries()

    sess = tf.Session(graph=graph)
    writer = tf.train.SummaryWriter('model', graph)
    sess.run(init)

    for i in range(100):
        input_tensor = np.random.randn(i)
        feed_dict = {a: input_tensor}
        _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=step)

    writer.flush()
    writer.close()
    sess.close()




