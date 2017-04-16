import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages

# Layers

def conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      # Xavier initialization
      kernel = tf.get_variable(
                'weights', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

def relu(name, x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def fully_connected(name, x, batch_size, out_dim):
    with tf.variable_scope(name):
        x = tf.reshape(x, [batch_size, -1])
        w = tf.get_variable(
             'weights', [x.get_shape()[1], out_dim],
             initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

def max_pool(name, x, filter_size, strides):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, [1, filter_size, filter_size, 1],
                              strides, padding='SAME')

def avg_pool(name, x, filter_size, strides):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(x, [1, filter_size, filter_size, 1],
                              strides, padding='SAME')

def batch_norm(name, x, training, extra_train_ops):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if training:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())

      return y

def softmax(name, x):
    with tf.variable_scope(name):
        return tf.nn.softmax(x)

# Common blocks
def residual(name, x, in_filters, out_filters, stride,
             training, extra_train_ops):

    orig_x = x
    with tf.variable_scope(name):
        x = batch_norm('bn1', x, training, extra_train_ops)
        x = relu('relu1', x)
        x = conv('conv1', x, 3, in_filters, out_filters, [1, stride, stride, 1])

        x = batch_norm('bn2', x, training, extra_train_ops)
        x = relu('relu2', x)
        x = conv('conv2', x, 3, out_filters, out_filters, [1, 1, 1, 1])

        if stride > 1:
            orig_x = avg_pool('down', orig_x, stride, [1, stride, stride, 1])
        if in_filters != out_filters:
            orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filters-in_filters)//2, (out_filters-in_filters)//2]])

        x += orig_x

    return x

def global_avg_pool(name, x):
    with tf.variable_scope(name):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
