import tensorflow as tf

if __name__=='__main__':
    g1 = tf.Graph()
    g2 = tf.Graph()

    with g1.as_default():
        a = tf.constant(2, name='input_a')
        b = tf.constant(3, name='input_b')
        c = tf.add(a, b, name='add_c')
        d = tf.mul(a, b, name='mul_d')
        e = tf.add(c, d, name='add_e')

    with g2.as_default():
        a2 = tf.constant([3, 4], name='tensor_input')
        b2 = tf.reduce_prod(a2, name='prod_b2')
        c2 = tf.reduce_sum(a2, name='sum_c2')
        d2 = tf.add(b2, c2, name='result')