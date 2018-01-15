import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import os
import random

from layers import *
def upsample_layer(image_batch, scale, scope_name):
    height, width = image_batch.shape[1:3]
    scale_h = int(height*scale)
    scale_w = int(width*scale)
    return tf.image.resize_nearest_neighbor(image_batch, [scale_h, scale_w], name = scope_name)

def build_ResnetBlock(inputres, dim, name = "resnet"):
    with tf.variable_scope(name):
        out_res = general_conv2d(inputres, dim, 3, 1, name = "conv1")
        out_res = general_conv2d(out_res, dim, 3, 1, name = "conv2")
        return tf.nn.relu(out_res + inputres)


def built_network(input_batch):
    with tf.name_scope("network") as scope:
        filters = 64
        #block 1 -- outputs 112x112x64
        conv1_1 = general_conv2d(input_batch, filters, 3, 1, name = "conv1_1")
        #conv1_2 = general_conv2d(conv1_1, filters, 3, 1, name = "conv1_2")
        pool_1 = max_pool(conv1_1, 2, 2, 2, 2, name = "pool_1")
        #pool_1 = general_conv2d(conv1_1, filters, 3, 2, name = "pool_1")
        #print(pool_1)

        #block2 -- outputs 56x56x128
        conv2_1 = general_conv2d(pool_1, filters * 2, 3, 1, name = "conv2_1")
        #conv2_2 = general_conv2d(conv2_1, filters * 2, 3, 1, name="conv2_2")
        pool_2 = max_pool(conv2_1, 2, 2, 2, 2, name = "pool_2")
        #pool_2 = general_conv2d(conv2_1, filters * 2, 3, 2, name = "pool_2")
        #print(pool_2)

        #block3 -- outputs 28x28x256
        conv3_1 = general_conv2d(pool_2, filters * 4, 3, 1, name = "conv3_1")
        conv3_2 = general_conv2d(conv3_1, filters * 4, 3, 1, name = "conv3_2")
        #conv3_3 = general_conv2d(conv3_2, filters * 4, 3, 1, name = "conv3_3")
        pool_3 = max_pool(conv3_2, 2, 2, 2, 2, name = "pool_3")
        #pool_3 = general_conv2d(conv3_2, filters * 4, 3, 2, name = "pool_3")
        #print(pool_3)

        #block4 -- outputs 14x14x512
        conv4_1 = general_conv2d(pool_3, filters * 8, 3, 1, name = "conv4_1")
        conv4_2 = general_conv2d(conv4_1, filters * 8, 3, 1, name="conv4_2")
        #conv4_3 = general_conv2d(conv4_2, filters * 8, 3, 1, name="conv4_3")
        pool_4 = max_pool(conv4_2, 2, 2, 2, 2, name = "pool_4")
        #pool_4 = general_conv2d(conv4_2, filters * 8, 3, 2, name = "pool_4")
        #print(pool_4)

        #block5 -- outputs 7x7x512
        conv5_1 = general_conv2d(pool_4, filters * 8, 3, 1, name="conv5_1")
        conv5_2 = general_conv2d(conv5_1, filters * 8, 3, 1, name="conv5_2")
        #conv5_3 = general_conv2d(conv5_2, filters * 8, 3, 1, name="conv5_3")
        pool_5 = max_pool(conv5_2, 2, 2, 2, 2, name="pool_5")
        #pool_5 = general_conv2d(conv5_2, filters * 8, 3, 2, name = "pool_5")
        #print(pool_5)

        # flatten
        shp = pool_5.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool_5, [-1, flattened_shape], name="resh1")
        #print(resh1)

        #fully connected
        FC_1 = layers.fully_connected(inputs=resh1,
                               num_outputs=4096,
                               scope='FC_1')
        FC_1_dropout = tf.nn.dropout(FC_1, keep_prob=0.5, name = "FC_1_dropout")

        #print(FC_1)

        FC_2 = layers.fully_connected(inputs=FC_1_dropout,
                               num_outputs=4096,
                               scope='FC_2')

        FC_2_dropout = tf.nn.dropout(FC_2, keep_prob=0.5, name="FC_2_dropout")

        #print(FC_2)

        FC_3 = layers.fully_connected(inputs=FC_2_dropout,
                                      num_outputs=21,
                                      scope='FC_3')
        #output = tf.nn.softmax(FC_3, name="output")
    return FC_3

'''def built_network2(input_batch):
    with tf.name_scope("network2") as scope:
        filters = 64
        kernel_size = 3
        # flatten
        shp = conv4_1.get_shape()
        flattened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(conv4_1, [-1, flattened_shape], name="resh1")
        print(resh1)

        # fully connected
        FC_1 = layers.fully_connected(inputs=resh1,
                                      num_outputs=4096,
                                      scope='FC_1')
        print(FC_1)

        FC_2 = layers.fully_connected(inputs=FC_1,
                                      num_outputs=4096,
                                      scope='FC_2')
        print(FC_2)

        FC_3 = layers.fully_connected(inputs=FC_2,
                                      num_outputs=21,
                                      scope='FC_3')
        print(FC_3)
    return FC_3
'''


# Loss函数
def L1_loss(out_batch, index_batch, name):
    with tf.name_scope(name) as scope:
        loss = tf.reduce_mean(tf.squared_difference(out_batch, index_batch))
        tf.summary.scalar(name, loss)
        return loss


#loss function
def whole_loss(output_batch, index_batch):
    with tf.name_scope('loss') as scope:
        whole_loss = L1_loss(output_batch, index_batch, name = "whole_loss")
        tf.summary.scalar("whole_loss", whole_loss)
        return whole_loss


# 训练操作
def training(loss, global_step):
    with tf.name_scope('OPTIMIZE') as scope:
        lr = tf.train.exponential_decay(1e-3, global_step, 10000, 0.7, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op