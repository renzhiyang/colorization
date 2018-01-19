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


def newEncode(input_batch):
    with tf.name_scope("newEncode") as scope:
        kernel_size = 3
        filters = 64
        conv1_1 = general_conv2d(input_batch, filters, kernel_size, 2, name = "newEn_conv11")
        conv1_2 = general_conv2d(conv1_1, filters * 2, kernel_size, 1, name = "newEn_conv12")
        conv2_1 = general_conv2d(conv1_2, filters * 2, kernel_size, 2, name = "newEn_conv21")
        conv2_2 = general_conv2d(conv2_1, filters * 4, kernel_size, 1, name = "newEn_conv22")
        conv3_1 = general_conv2d(conv2_2, filters * 4, kernel_size, 2, name = "newEn_conv31")
        conv3_2 = general_conv2d(conv3_1, filters * 8, kernel_size, 1, name = "newEn_conv32")
        return [conv1_1, conv2_1], conv3_2

def newMiddle_layer(input_batch):
    with tf.name_scope("newMiddle_layer") as scope:
        kernel_size = 3
        filters = 64
        conv1 = general_conv2d(input_batch, filters * 8, kernel_size, 1, name = "newMid_conv1")
        conv2 = general_conv2d(conv1, filters * 8, kernel_size, 1, name = "newMid_conv2")
        conv3 = general_conv2d(conv2, filters * 8, kernel_size, 1, name = "newMid_conv3")
        conv4 = general_conv2d(conv3, filters * 8, kernel_size, 1, name="newMid_conv4")
        conv5 = general_conv2d(conv4, filters * 8, kernel_size, 1, name="newMid_conv5")
        conv6 = general_conv2d(conv5, filters * 8, kernel_size, 1, name="newMid_conv6")
        conv7 = general_conv2d(conv6, filters * 8, kernel_size, 1, name="newMid_conv7")
        #conv1 = build_ResnetBlock(input_batch, filters * 8, name="newMid_conv1")
        #conv2 = build_ResnetBlock(conv1, filters * 8, name="newMid_conv2")
        #conv3 = build_ResnetBlock(conv2, filters * 8, name="newMid_conv3")
        #conv4 = build_ResnetBlock(conv3, filters * 8, name="newMid_conv4")
        #conv5 = build_ResnetBlock(conv4, filters * 8, name="newMid_conv5")
        #conv6 = build_ResnetBlock(conv5, filters * 8, name="newMid_conv6")
        conv8 = general_conv2d(conv7, filters * 4, kernel_size, 1, name = "newMid_conv8")
        return conv8

def newDecode(input_batch, uNetLayer):
    with tf.name_scope("newEncode2") as scope:
        kernel_size = 3
        filters = 64
        conv1 = general_conv2d(input_batch, filters * 2, kernel_size, 1, name = "newDeco2_conv1")
        coef1 = tf.get_variable("coef1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        uplayer1 = upsample_layer(conv1, 2, scope_name="newDeco2_uplayer1") + (1 - coef1) * uNetLayer[1]
        conv2_1 = general_conv2d(uplayer1, filters, kernel_size, 1, name = "newDeco2_conv21")
        conv2_2 = general_conv2d(conv2_1, filters, kernel_size, 1, name = "newDeco2_conv22")
        coef2 = tf.get_variable("coef2", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        uplayer2 = upsample_layer(conv2_2, 2, scope_name="newDeco2_uplayer2") + (1 - coef2) * uNetLayer[0]
        conv3_1 = general_conv2d(uplayer2, filters / 2, kernel_size, 1, name = "newDeco2_conv31")
        conv3_2 = general_conv2d(conv3_1, 2, kernel_size, 1, do_relu=False, name = "newDeco2_conv32")
        conv3_2 = tf.nn.sigmoid(conv3_2, "addSigmoid")
        uplayer3 = upsample_layer(conv3_2, 2, scope_name = "newDeco2_uplayer3")
        return uplayer3

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

def built_network(input_ab_batch, sparse_input):
    with tf.name_scope("built_newwork") as scope:
        input_ab_batch = general_conv2d(input_ab_batch, 32, 3, 1, name="pre_conv_input")
        sparse_input = general_conv2d(sparse_input, 32, 3, 1, name="pre_conv_sparse")
        coef = tf.get_variable("coef", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        input_batch = input_ab_batch + (1 - coef) * sparse_input

        unetLayer, encodeResult = newEncode(input_batch)
        middle_output = newMiddle_layer(encodeResult)
        #theme_output = theme_features_network(theme_input, middle_output.shape[-1].value)
        #fusion_out = fusion_layer(middle_output, theme_output)
        out_ab_batch = newDecode(middle_output, unetLayer)
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

def get_sobel_batch(image_batch):
    assert image_batch.shape[-1].value == 2
    batch_size = image_batch.shape[0].value
    height = image_batch.shape[1].value
    width = image_batch.shape[2].value

    batch1 = image_batch[:, :, :, 0]
    batch2 = image_batch[:, :, :, 1]
    batch = tf.reshape(tf.concat([batch1, batch2], 0), [batch_size*2, height, width, 1])
    batch_sobel = sobel2(batch)
    batch_sobel = tf.concat([batch_sobel[0], batch_sobel[1]], 0)

    return batch_sobel

#loss function
def whole_loss(output_ab_batch, index_ab_batch, image_ab_batch, mask2channels):
    with tf.name_scope('loss') as scope:
        image_exceptPoints = image_ab_batch - image_ab_batch * mask2channels
        out_exceptPoints = output_ab_batch - output_ab_batch * mask2channels
        local_output_ab = output_ab_batch * mask2channels
        local_index_ab = index_ab_batch * mask2channels

        #local loss, do gradient between output and index
        #sobel_loss = sobeled_losses(output_ab_batch, index_ab_batch)
        gra_out_ab = get_sobel_batch(out_ab_batch)
        gra_index_ab = get_sobel_batch(index_ab_batch)
        sobel_loss = tf.reduce_mean(tf.square(gra_out_ab - gra_index_ab))

        localpoint_loss = L1_loss(local_output_ab, local_index_ab, name = "localPoint_loss")
        index_loss = tf.losses.huber_loss(output_ab_batch, index_ab_batch)
        whole_loss = index_loss + sobel_loss

        tf.summary.scalar("whole_loss", whole_loss)
        tf.summary.scalar("localPoint_loss", localpoint_loss)
        tf.summary.scalar("sobel_loss", sobel_loss)
        return whole_loss, [index_loss, localpoint_loss, sobel_loss]

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

