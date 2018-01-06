import tensorflow as tf
import numpy as np

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

#leaky relu
def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

#old batch normalize
def batch_norm_layer(x, training = tf.constant(True)):
    with tf.variable_scope("batch_norm"):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        #global
        #axises = np.arange(len(x.shape) - 1)
        axises = [0]
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


#instance normalize
def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out


def general_conv2d(inputconv, filters, kernel_size, strides, do_norm=False, do_relu=True, relufactor=0.2, name="conv2d"):
    stddev=0.02
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(inputs=inputconv,
                                       num_outputs=filters,
                                       kernel_size=kernel_size,
                                       stride=strides,
                                       activation_fn = None,
                                       #weights_initializer = tf.truncated_normal_initializer(stddev=stddev),
                                       scope=name)

        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                #conv = lrelu(conv, relufactor, "lrelu")
                conv = tf.nn.tanh(conv, name = "tanh")
        return conv


def general_deconv2d(inputconv, filters, kernel_size, strides, padding = "SAME", name = "conv2d", do_norm=True, do_relu=True, relufactor=0):
    stddev = 0.02
    with tf.variable_scope(name):

        conv = tf.layers.conv2d_transpose(inputs = inputconv, 
                                        filters = filters, 
                                        kernel_size = kernel_size, 
                                        kernel_initializer = tf.truncated_normal_initializer(stddev=stddev),
                                        strides = strides, 
                                        padding = padding)

        if do_norm:
            conv = instance_norm(conv)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv