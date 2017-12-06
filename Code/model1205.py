import tensorflow as tf
import numpy as np
import os
import random

from layers import *


def build_ResnetBlock(inputres, dim, name = "resnet"):
    with tf.variable_scope(name):
        out_res = general_conv2d(inputres, dim, 3, 1, name = "conv1")
        out_res = general_conv2d(out_res, dim, 3, 1, name = "conv2")
        return tf.nn.relu(out_res + inputres)


def built_network(input_ab_batch, sparse_batch, mask_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64
        # input_batch = 224*224*5
        input_batch = tf.concat([input_ab_batch, sparse_batch, mask_batch], 3)

        # conv1_1 = 224*224*64
        conv1_1 = general_conv2d(input_batch, filters, kernel_size, 1, name="conv1_1")
        # conv1_2 = 224*224*64
        conv1_2 = general_conv2d(conv1_1, filters, kernel_size, 1, name="conv1_2")
        # conv2_1 = 112*112*128
        conv2_1 = general_conv2d(conv1_2, filters * 2, kernel_size, 2, name="conv2_1")
        # conv2_2 = 112*112*128
        conv2_2 = general_conv2d(conv2_1, filters * 2, kernel_size, 1, name="conv2_2")
        # conv3_1 = 56*56*256
        conv3_1 = general_conv2d(conv2_2, filters * 4, kernel_size, 2, name="conv3_1")
        # conv3_2 = 56*56*256
        conv3_2 = general_conv2d(conv3_1, filters * 4, kernel_size, 1, name="conv3_2")
        # conv3_3 = 56*56*256
        conv3_3 = general_conv2d(conv3_2, filters * 4, kernel_size, 1, name="conv3_3")

        # middle level 50*28*28*512 , 建立四个resnet的block
        conv4 = general_conv2d(conv3_3, filters * 8, kernel_size, 2, name="conv4")
        conv4_1 = build_ResnetBlock(conv4, filters * 8, name="conv4_1")
        conv4_2 = build_ResnetBlock(conv4_1, filters * 8, name="conv4_2")
        conv4_3 = build_ResnetBlock(conv4_2, filters * 8, name="conv4_3")
        conv4_4 = build_ResnetBlock(conv4_3, filters * 8, name="conv4_4")


        # conv5_1 = 56*56*256
        coef1 = tf.get_variable("coef1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv5_1 = general_deconv2d(conv4_4, filters * 4, kernel_size, 2, name="conv5_1") + (1 - coef1) * conv3_3
        # conv5_2 = 56*56*256
        conv5_2 = general_conv2d(conv5_1, filters * 4, kernel_size, 1, name="conv5_2")
        # conv5_3 = 56*56*256
        conv5_3 = general_conv2d(conv5_2, filters * 4, kernel_size, 1, name="conv5_3")
        # conv6_1 = 112*112*128
        coef2 = tf.get_variable("coef2", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv6_1 = general_deconv2d(conv5_3, filters * 2, kernel_size, 2, name="conv6_1") + (1 - coef2) * conv2_2
        # conv6_2 = 112*112*128
        conv6_2 = general_conv2d(conv6_1, filters * 2, kernel_size, 1, name="conv6_2")
        # conv7_1 = 224*224*64
        coef3 = tf.get_variable("coef3", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv7_1 = general_deconv2d(conv6_2, filters, kernel_size, 2, name="conv7_1") + (1 - coef3) * conv1_2
        # conv7_2 = 224*224*64
        conv7_2 = general_conv2d(conv7_1, filters, kernel_size, 1, name="conv7_2")

        # conv8_1 = 224*224*2
        conv8_1 = general_conv2d(conv7_2, 2, 1, 1, name="conv8_1")

        return tf.nn.tanh(conv8_1, name="output")



# Loss函数
def L1_loss(out_batch, index_batch, name):
    with tf.name_scope(name) as scope:
        loss = tf.reduce_mean(tf.squared_difference(out_batch, index_batch))
        tf.summary.scalar(name, loss)
        return loss

def getResult_sobel(input_batch):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x_one = tf.nn.conv2d(input_batch[:, :, :, 0:1], sobel_x_filter,
                                   strides=[1, 1, 1, 1], padding='SAME')
    filtered_x_two = tf.nn.conv2d(input_batch[:, :, :, 1:2], sobel_x_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')
    filtered_y_one = tf.nn.conv2d(input_batch[:, :, :, 0:1], sobel_y_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')
    filtered_y_two = tf.nn.conv2d(input_batch[:, :, :, 1:2], sobel_y_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')

    return filtered_x_one, filtered_x_two, filtered_y_one, filtered_y_two

#对输出的图片和目标图之间做梯度loss
def sobeled_losses(output_batch, index_batch, name = "sobeled_loss"):
    with tf.variable_scope(name):
        out_x_a, out_x_b, out_y_a, out_y_b =  getResult_sobel(output_batch)
        index_x_a, index_x_b, index_y_a, index_y_b = getResult_sobel(index_batch)
        loss_x_a = L1_loss(out_x_a, index_x_a, "loss_x_a")
        loss_x_b = L1_loss(out_x_b, index_x_b, "loss_x_b")
        loss_y_a = L1_loss(out_y_a, index_y_a, "loss_y_a")
        loss_y_b = L1_loss(out_y_b, index_y_b, "loss_y_b")
        loss = loss_x_a + loss_x_b + loss_y_a + loss_y_b
        return loss

#输出的图片中与sparse中相对应的点，与sparse做loss，也就是mask loss
def mask_losses(output_batch, mask_batch, sparse_batch, name = "mask_loss"):
    with tf.variable_scope(name):
        a_batch = tf.multiply(output_batch[:, :, :, 0:1], mask_batch)
        b_batch = tf.multiply(output_batch[:, :, :, 1:2], mask_batch)
        outpoint_batch = tf.concat([a_batch, b_batch], 3)
        mask_loss = L1_loss(output_batch, sparse_batch, name = "mask_loss")
        return mask_loss



#总的loss，供外部调用
def whole_loss(output_batch, index_batch, mask_batch, sparse_batch):
    with tf.name_scope('whole_loss') as scope:
        sobeled_loss = sobeled_losses(output_batch, index_batch)
        mask_loss = mask_losses(output_batch, mask_batch, sparse_batch)
        loss = sobeled_loss + mask_loss
        return loss


def get_PSNR(out_ab_batch, index_ab_batch):
    b = 8
    MAX = 2 ** b - 1
    RMSE = tf.sqrt(tf.reduce_mean(tf.squared_difference(out_ab_batch, index_ab_batch)))
    PSNR = 10 * log10( MAX / RMSE)
    tf.summary.scalar('RMSE', RMSE)
    tf.summary.scalar('PSNR', PSNR)
    return RMSE, PSNR

# 训练操作
def training(loss, global_step):
    with tf.name_scope('OPTIMIZE') as scope:
        lr = tf.train.exponential_decay(1e-3, global_step, 10000, 0.7, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
