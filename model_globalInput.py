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


def encode(input_batch):
    with tf.name_scope("encode") as scope:
        kernel_size = 3
        filters = 64
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
        return conv1_2, conv2_2, conv3_3

def middle_layer(input_batch):
    with tf.name_scope("middle_layer") as scope:
        kernel_size = 3
        filters = 64
        conv1 = general_conv2d(input_batch, filters * 8, kernel_size, 2, name="conv1")
        conv2 = build_ResnetBlock(conv1, filters * 8, name="conv2")
        conv3 = build_ResnetBlock(conv2, filters * 8, name="conv3")
        conv4 = build_ResnetBlock(conv3, filters * 8, name="conv4")
        #conv5 = build_ResnetBlock(conv4, filters * 8, name="conv5")
        return conv4

def decode(input_batch, layer1, layer2, layer3):
    with tf.name_scope("decode") as scope:
        kernel_size = 3
        filters = 64
        # conv5_1 = 56*56*256
        conv5_1 = upsample_layer(input_batch, 2, scope_name="conv5_1")
        conv5_1 = tf.concat([conv5_1, layer3], 3)
        # conv5_2 = 56*56*256
        conv5_2 = general_conv2d(conv5_1, filters * 4, kernel_size, 1, name="conv5_2")
        # conv5_3 = 56*56*256
        conv5_3 = general_conv2d(conv5_2, filters * 4, kernel_size, 1, name="conv5_3")
        # conv6_1 = 112*112*128
        conv6_1 = upsample_layer(conv5_3, 2, scope_name="conv6_1")
        conv6_1 = tf.concat([conv6_1, layer2], 3)
        # conv6_2 = 112*112*128
        conv6_2 = general_conv2d(conv6_1, filters * 2, kernel_size, 1, name="conv6_2")
        # conv7_1 = 224*224*128
        conv7_1 = upsample_layer(conv6_2, 2, scope_name="conv7_1")
        conv7_1 = tf.concat([conv7_1, layer1], 3)
        # conv7_2 = 224*224*128
        conv7_2 = general_conv2d(conv7_1, filters * 2, kernel_size, 1, name="conv7_2")
        conv7_3 = general_conv2d(conv7_2, filters * 2, kernel_size, 1, name="conv7_3")
        # conv8_1 = 224*224*2
        conv8_1 = general_conv2d(conv7_3, 2, 1, 1, name="conv8_1")
        return tf.nn.tanh(conv8_1, name="output")

def theme_features_network(theme_batch, num_outputs):
    with tf.name_scope("theme_features_network") as scope:
        batch_size = theme_batch.get_shape()[0].value
        theme_input = tf.reshape(theme_batch, [batch_size, -1])

        h_fc1 = layers.fully_connected(inputs=theme_input,
                                       num_outputs=512,
                                       scope=scope + '/fc1')
        h_fc2 = layers.fully_connected(inputs=h_fc1,
                                       num_outputs=512,
                                       scope=scope + '/fc2')

        h_fc3 = layers.fully_connected(inputs=h_fc2,
                                       num_outputs=num_outputs,
                                       scope=scope + '/fc3')

        return h_fc3

def fusion_layer(source_feature, target_feature):
    with tf.name_scope("fusion_layer") as scope:
        batch_size = source_feature.shape[0].value
        feature_channels = source_feature.shape[-1].value

        target_feature = tf.reshape(target_feature, [batch_size, 1, 1, feature_channels])

        coef = tf.get_variable(scope + 'coef',
                               shape=[1],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.5))
        fusion_feature = source_feature + (1 - coef) * target_feature
        print([scope, fusion_feature])

        return fusion_feature

def built_network(input_ab_batch, theme_input):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64
        # input_batch = 224*224*3
        layer1, layer2, layer3 = encode(input_ab_batch)
        middle_output = middle_layer(layer3)
        theme_output = theme_features_network(theme_input, middle_output.shape[-1].value)
        fusion_out = fusion_layer(middle_output, theme_output)
        out_ab_batch = decode(fusion_out, layer1, layer2, layer3)
        return out_ab_batch



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
def mask_losses(output_batch, mask_batch_2channels, sparse_batch, name = "mask_loss"):
    with tf.variable_scope(name):
        outpoint_batch = mask_batch_2channels * output_batch
        mask_loss = L1_loss(outpoint_batch, sparse_batch, name = "mask_L1_loss")
        return mask_loss


#loss function
def whole_loss(output_ab_batch, index_ab_batch, themeIndex_ab_batch, image_ab_batch):
    with tf.name_scope('loss') as scope:
        #index sobel loss
        #sobel_loss = sobeled_losses(output_ab_batch, index_ab_batch)
        #index loss
        index_loss = tf.losses.huber_loss(output_ab_batch, index_ab_batch, delta = 0.5)
        #image loss
        image_loss = tf.losses.huber_loss(output_ab_batch, image_ab_batch, delta = 0.5)
        #color theme loss
        color_loss = tf.losses.huber_loss(output_ab_batch, themeIndex_ab_batch, delta = 0.5)

        whole_loss = 0.1 * image_loss + 0.9 * (0.3 * index_loss + 0.7 * color_loss)
        tf.summary.scalar("whole_loss", whole_loss)
        tf.summary.scalar("image_loss", image_loss)
        tf.summary.scalar("index_loss", index_loss)
        tf.summary.scalar("color_loss", color_loss)
        return whole_loss, image_loss, index_loss, color_loss

def get_PSNR(out_ab_batch, index_ab_batch):
    #b = 8
    #MAX = 2 ** b - 1
    #RMSE = tf.sqrt(tf.reduce_mean(tf.squared_difference(out_ab_batch, index_ab_batch)))
    #PSNR = 10 * log10( MAX / RMSE)
    #tf.summary.scalar('RMSE', RMSE)
    #tf.summary.scalar('PSNR', PSNR)

    MSE = tf.reduce_mean(tf.square(out_ab_batch - index_ab_batch))
    PSNR = 10 * tf.log(1 / MSE) / np.log(10)
    tf.summary.scalar('MSE', MSE)
    tf.summary.scalar('PSNR', PSNR)
    return MSE, PSNR

# 训练操作
def training(loss, global_step):
    with tf.name_scope('OPTIMIZE') as scope:
        lr = tf.train.exponential_decay(1e-3, global_step, 10000, 0.7, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
