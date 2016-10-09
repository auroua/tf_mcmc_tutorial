import tensorflow as tf

if __name__ == '__main__':
    # graph = tf.get_default_graph()
    # # input_value = tf.constant(1.0)
    # # weight = tf.Variable(0.8)
    # # output_value = weight*input_value
    # operations = graph.get_operations()
    # print operations[0].node_def
    # # print sess.run(input_value)
    # for op in graph.get_operations():
    #     print op.name
    # print '-------------------'
    # op = graph.get_operations()[-1]
    # for op_input in op.inputs:
    #     print op_input

    # x = tf.constant(1.0, name='input')
    # y_ = tf.constant(0.0, name='label')
    # w = tf.Variable(0.8, name='weight')
    # y = tf.mul(w, x, name='output')
    # loss = tf.square((y-y_), name='loss')
    # optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
    # grads_and_vars = optim.compute_gradients(loss)
    #
    # init = tf.initialize_all_variables()
    # sess = tf.Session()
    # sess.run(init)
    # summary_y = tf.scalar_summary('output', y)
    #
    # summary_writer = tf.train.SummaryWriter('model', sess.graph)
    # print sess.run(grads_and_vars[0][0])
    #
    # sess.run(optim.apply_gradients(grads_and_vars))
    # # 0.76  0.8-1.6 * 0.025   learning rate is 0.025
    # print sess.run(w)
    #
    # train_step = optim.minimize(loss)
    # for i in range(100):
    #     summary_str = sess.run(summary_y)
    #     summary_writer.add_summary(summary_str, i)
    #     # print('before step {}, y is {}'.format(i, sess.run(y)))
    #     sess.run(train_step)

    x = tf.constant(1.0, name='input')
    w = tf.Variable(0.8, name='weight')
    y = tf.mul(w, x, name='output')
    y_ = tf.constant(0.0, name='correct_value')
    loss = tf.pow(y - y_, 2, name='loss')
    train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

    for value in [x, w, y, y_, loss]:
        tf.scalar_summary(value.op.name, value)

    summaries = tf.merge_all_summaries()

    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter('model', sess.graph)

    sess.run(tf.initialize_all_variables())
    for i in range(100):
        summary_writer.add_summary(sess.run(summaries), i)
        sess.run(train_step)
