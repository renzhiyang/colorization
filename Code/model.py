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


def upsample_layer(image_batch, scale, scope_name):
    height, width = image_batch.shape[1:3]
    scale_h = int(height*scale)
    scale_w = int(width*scale)
    return tf.image.resize_nearest_neighbor(image_batch, [scale_h, scale_w], name = scope_name)

#the full network
def built_network1122(image_batch, sparse_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64
        #input_batch = 224*224*6
        input_batch = tf.concat([image_batch, sparse_batch],3)
        
        #conv1_1 = 224*224*64
        conv1_1 = general_conv2d(input_batch, filters, kernel_size, 1, name = "conv1_1")
        #conv1_2 = 224*224*64
        conv1_2 = general_conv2d(conv1_1, filters, kernel_size, 1, name = "conv1_2")
        #conv2_1 = 112*112*128
        conv2_1 = general_conv2d(conv1_2, filters*2, kernel_size, 2, name = "conv2_1")
        #conv2_2 = 112*112*128
        conv2_2 = general_conv2d(conv2_1, filters*2, kernel_size, 1, name = "conv2_2")
        #conv3_1 = 56*56*256
        conv3_1 = general_conv2d(conv2_2, filters*4, kernel_size, 2, name = "conv3_1")
        #conv3_2 = 56*56*256
        conv3_2 = general_conv2d(conv3_1, filters*4, kernel_size, 1, name = "conv3_2")
        #conv3_3 = 56*56*256
        conv3_3 = general_conv2d(conv3_2, filters*4, kernel_size, 1, name = "conv3_3")

        # middle level 50*28*28*512 , 建立四个resnet的block
        conv4 = general_conv2d(conv3_3, filters * 8, kernel_size, 2, name="conv4")
        conv4_1 = build_ResnetBlock(conv4, filters * 8, name="conv4_1")
        conv4_2 = build_ResnetBlock(conv4_1, filters * 8, name="conv4_2")
        conv4_3 = build_ResnetBlock(conv4_2, filters * 8, name="conv4_3")
        conv4_4 = build_ResnetBlock(conv4_3, filters * 8, name="conv4_4")

        #conv5_1 = 56*56*256
        coef1 = tf.get_variable("coef1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv5_1 = general_deconv2d(conv4_4, filters*4, kernel_size, 2, name = "conv5_1") + (1 - coef1) * conv3_3
        #conv5_2 = 56*56*256
        conv5_2 = general_conv2d(conv5_1, filters*4, kernel_size, 1, name = "conv5_2")
        #conv5_3 = 56*56*256
        conv5_3 = general_conv2d(conv5_2, filters*4, kernel_size, 1, name = "conv5_3")
        #conv6_1 = 112*112*128
        coef2 = tf.get_variable("coef2", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv6_1 = general_deconv2d(conv5_3, filters*2, kernel_size, 2, name = "conv6_1") + (1 - coef2) * conv2_2
        #conv6_2 = 112*112*128
        conv6_2 = general_conv2d(conv6_1, filters*2, kernel_size, 1, name = "conv6_2")
        #conv7_1 = 224*224*64
        coef3 = tf.get_variable("coef3", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv7_1 = general_deconv2d(conv6_2, filters, kernel_size, 2, name = "conv7_1") + (1 - coef3) * conv1_2
        #conv7_2 = 224*224*64
        conv7_2 = general_conv2d(conv7_1, filters, kernel_size, 1, name = "conv7_2")

        #conv8_1 = 224*224*2
        conv8_1 = general_conv2d(conv7_2, 2, kernel_size, 1, name = "conv8_1")

        return tf.nn.tanh(conv8_1, name="output")


def built_network1123(image_l_batch, sparse_ab_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64
        # input_batch = 224*224*3
        input_batch = tf.concat([image_l_batch, sparse_ab_batch], 3)

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

        # middle level 50*28*28*512
        conv4_1 = general_conv2d(conv3_3, filters * 8, kernel_size, 2, name="conv4_1")
        conv4_2 = general_conv2d(conv4_1, filters * 8, kernel_size, 1, name="conv4_2")
        conv4_3 = general_conv2d(conv4_2, filters * 8, kernel_size, 1, name="conv4_3")
        conv4_4 = general_conv2d(conv4_3, filters * 8, kernel_size, 1, name="conv4_4")
        conv4_5 = general_conv2d(conv4_4, filters * 8, kernel_size, 1, name="conv4_5")
        conv4_6 = general_conv2d(conv4_5, filters * 8, kernel_size, 1, name="conv4_6")
        conv4_7 = general_conv2d(conv4_6, filters * 8, kernel_size, 1, name="conv4_7")
        conv4_8 = general_conv2d(conv4_7, filters * 8, kernel_size, 1, name="conv4_8")
        conv4_9 = general_conv2d(conv4_8, filters * 8, kernel_size, 1, name="conv4_9")
        conv4_10 = general_conv2d(conv4_9, filters * 8, kernel_size, 1, name="conv4_10")
        conv4_11 = general_conv2d(conv4_10, filters * 8, kernel_size, 1, name="conv4_11")
        conv4_12 = general_conv2d(conv4_11, filters * 8, kernel_size, 1, name="conv4_12")


        # conv5_1 = 56*56*256
        conv5_1 = general_deconv2d(conv4_4, filters * 4, kernel_size, 2, name="conv5_1") + conv3_3
        # conv5_2 = 56*56*256
        conv5_2 = general_conv2d(conv5_1, filters * 4, kernel_size, 1, name="conv5_2")
        # conv5_3 = 56*56*256
        conv5_3 = general_conv2d(conv5_2, filters * 4, kernel_size, 1, name="conv5_3")
        # conv6_1 = 112*112*128
        conv6_1 = general_deconv2d(conv5_3, filters * 2, kernel_size, 2, name="conv6_1") + conv2_2
        # conv6_2 = 112*112*128
        conv6_2 = general_conv2d(conv6_1, filters * 2, kernel_size, 1, name="conv6_2")
        # conv7_1 = 224*224*64
        conv7_1 = general_deconv2d(conv6_2, filters, kernel_size, 2, name="conv7_1") + conv1_2
        # conv7_2 = 224*224*64
        conv7_2 = general_conv2d(conv7_1, filters, kernel_size, 1, name="conv7_2")

        # conv8_1 = 224*224*2
        conv8_1 = general_conv2d(conv7_2, 2, 1, 1, name="conv8_1")

        return tf.nn.tanh(conv8_1, name="output")


def built_network1124(image_batch, sparse_ab_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64

        #sparse_conv1_1 = 224*224*64
        sparse_conv1_1 = general_conv2d(sparse_ab_batch, filters, kernel_size, 1, name = "sparse_conv1_1")
        # sparse_conv1_2 = 224*224*64
        sparse_conv1_2 = general_conv2d(sparse_conv1_1, filters, kernel_size, 1, name="sparse_conv1_2")
        # sparse_conv2_1 = 112*112*128
        sparse_conv2_1 = general_conv2d(sparse_conv1_2, filters * 2, kernel_size, 2, name="sparse_conv2_1")
        # sparse_conv2_2 = 112*112*128
        sparse_conv2_2 = general_conv2d(sparse_conv2_1, filters * 2, kernel_size, 1, name="sparse_conv2_2")
        # sparse_conv3_1 = 56*56*256
        sparse_conv3_1 = general_conv2d(sparse_conv2_2, filters * 4, kernel_size, 2, name="sparse_conv3_1")

        # input_batch = 224*224*5
        input_batch = tf.concat([image_batch, sparse_ab_batch], 3)
        # conv1_1 = 224*224*64
        conv1_1 = general_conv2d(input_batch, filters, kernel_size, 1, name="conv1_1")
        # conv1_2 = 224*224*64
        conv1_2 = general_conv2d(conv1_1, filters, kernel_size, 1, name="conv1_2")
        # conv2_1 = 112*112*128
        coef2_1 = tf.get_variable("coef2_1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv2_1 = general_conv2d(conv1_2, filters * 2, kernel_size, 2, name="conv2_1") + (1 - coef2_1) * sparse_conv2_1
        # conv2_2 = 112*112*128
        conv2_2 = general_conv2d(conv2_1, filters * 2, kernel_size, 1, name="conv2_2")
        # conv3_1 = 56*56*256
        coef3_1 = tf.get_variable("coef3_1", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv3_1 = general_conv2d(conv2_2, filters * 4, kernel_size, 2, name="conv3_1") + (1 - coef3_1) * sparse_conv3_1
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
        conv8_1 = general_conv2d(conv7_2, 2, kernel_size, 1, name="conv8_1")

        return tf.nn.tanh(conv8_1, name="output")


def built_network1126(image_batch, sparse_ab_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64

        # sparse_conv1_1 = 224*224*64
        sparse_conv1_1 = general_conv2d(sparse_ab_batch, filters, kernel_size, 1, name="sparse_conv1_1")
        # sparse_conv1_2 = 224*224*64
        sparse_conv1_2 = general_conv2d(sparse_conv1_1, filters, kernel_size, 1, name="sparse_conv1_2")
        # sparse_conv2_1 = 112*112*128
        sparse_conv2_1 = general_conv2d(sparse_conv1_2, filters * 2, kernel_size, 2, name="sparse_conv2_1")
        # sparse_conv2_2 = 112*112*128
        sparse_conv2_2 = general_conv2d(sparse_conv2_1, filters * 2, kernel_size, 1, name="sparse_conv2_2")
        # sparse_conv3_1 = 56*56*256
        sparse_conv3_1 = general_conv2d(sparse_conv2_2, filters * 4, kernel_size, 2, name="sparse_conv3_1")

        # image_batch = 224*224*3，将image和sparse的通道值相加，不用concat
        #image_batch[:, :, :, 1] += sparse_ab_batch[:, :, :, 0]
        #image_batch[:, :, :, 2] += sparse_ab_batch[:, :, :, 1]
        image_batch = tf.concat([image_batch, sparse_ab_batch], 3)
        # conv1_1 = 224*224*64
        conv1_1 = general_conv2d(image_batch, filters, kernel_size, 1, name="conv1_1")
        # conv1_2 = 224*224*64
        conv1_2 = general_conv2d(conv1_1, filters, kernel_size, 1, name="conv1_2")
        # conv2_1 = 112*112*128
        conv2_1 = general_conv2d(conv1_2, filters * 2, kernel_size, 2, name="conv2_1") + sparse_conv2_1
        # conv2_2 = 112*112*128
        conv2_2 = general_conv2d(conv2_1, filters * 2, kernel_size, 1, name="conv2_2")
        # conv3_1 = 56*56*256
        conv3_1 = general_conv2d(conv2_2, filters * 4, kernel_size, 2, name="conv3_1") + sparse_conv3_1
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
        conv8_1 = general_conv2d(conv7_2, 2, kernel_size, 1, name="conv8_1")

    return tf.nn.tanh(conv8_1, name="output")


def built_network1127(image_batch, sparse_ab_batch):
    with tf.name_scope("network") as scope:
        kernel_size = 3
        filters = 64

        # sparse_conv1_1 = 224*224*64
        sparse_conv1_1 = general_conv2d(sparse_ab_batch, filters, kernel_size, 1, name="sparse_conv1_1")
        # sparse_conv1_2 = 224*224*64
        sparse_conv1_2 = general_conv2d(sparse_conv1_1, filters, kernel_size, 1, name="sparse_conv1_2")
        # sparse_conv2_1 = 112*112*128
        sparse_conv2_1 = general_conv2d(sparse_conv1_2, filters * 2, kernel_size, 2, name="sparse_conv2_1")
        # sparse_conv2_2 = 112*112*128
        sparse_conv2_2 = general_conv2d(sparse_conv2_1, filters * 2, kernel_size, 1, name="sparse_conv2_2")
        # sparse_conv3_1 = 56*56*256
        sparse_conv3_1 = general_conv2d(sparse_conv2_2, filters * 4, kernel_size, 2, name="sparse_conv3_1")

        # input_batch = 224*224*3，将image和sparse的ab通道值相加
        l_batch = image_batch[:, :, :, :1]
        ab_batch = image_batch[:, :, :, 1:]
        ab_batch = ab_batch + sparse_ab_batch
        input_batch = tf.concat([l_batch, ab_batch], 3)
        # input_batch = 224*224*5，用concat将两个合并
        #input_batch = tf.concat([image_batch, sparse_ab_batch], 3)

        # conv1_1 = 224*224*64
        conv1_1 = general_conv2d(input_batch, filters, kernel_size, 1, name="conv1_1")
        # conv1_2 = 224*224*64
        conv1_2 = general_conv2d(conv1_1, filters, kernel_size, 1, name="conv1_2")
        # conv2_1 = 112*112*128
        conv2_1 = general_conv2d(conv1_2, filters * 2, kernel_size, 2, name="conv2_1") + sparse_conv2_1
        # conv2_2 = 112*112*128
        conv2_2 = general_conv2d(conv2_1, filters * 2, kernel_size, 1, name="conv2_2")
        # conv3_1 = 56*56*256
        conv3_1 = general_conv2d(conv2_2, filters * 4, kernel_size, 2, name="conv3_1") + sparse_conv3_1
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
        upLayer_1 = general_deconv2d(conv5_1, 2, kernel_size, 1, name="upLayer_1")  #56*56*2

        # conv5_2 = 56*56*256
        conv5_2 = general_conv2d(conv5_1, filters * 4, kernel_size, 1, name="conv5_2")
        # conv5_3 = 56*56*256
        conv5_3 = general_conv2d(conv5_2, filters * 4, kernel_size, 1, name="conv5_3")
        # conv6_1 = 112*112*128
        coef2 = tf.get_variable("coef2", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv6_1 = general_deconv2d(conv5_3, filters * 2, kernel_size, 2, name="conv6_1") + (1 - coef2) * conv2_2
        upLayer_2 = general_deconv2d(conv6_1, 2, kernel_size, 1, name="upLayer_2")  #112*112*2

        # conv6_2 = 112*112*128
        conv6_2 = general_conv2d(conv6_1, filters * 2, kernel_size, 1, name="conv6_2")
        # conv7_1 = 224*224*64
        coef3 = tf.get_variable("coef3", shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
        conv7_1 = general_deconv2d(conv6_2, filters, kernel_size, 2, name="conv7_1") + (1 - coef3) * conv1_2
        #upLayer_3 = general_deconv2d(conv7_1, 2, kernel_size, 1, name="upLayer_3")  # 224*224*2

        # conv7_2 = 224*224*64
        conv7_2 = general_conv2d(conv7_1, filters, kernel_size, 1, name="conv7_2")

        # conv8_1 = 224*224*2
        conv8_1 = general_conv2d(conv7_2, 2, kernel_size, 1, name="conv8_1")
        output = tf.nn.tanh(conv8_1, name="output")

    return output, upLayer_1, upLayer_2

def bilinear_of_indexbatch(index_batch):
    sp = index_batch.get_shape()[1].value
    layer2 = tf.image.resize_bilinear(index_batch, (sp//2, sp//2)) #112*112*2
    layer1 = tf.image.resize_bilinear(index_batch, (sp//4, sp//4)) #112*112*2
    return layer1, layer2


# Loss函数
def losses1122(out_image_batch, ab_image_batch, index_image_batch):
    with tf.name_scope("LOSS") as scope:
        loss1 = tf.reduce_mean(tf.squared_difference(out_image_batch, ab_image_batch))
        loss2 = tf.reduce_mean(tf.squared_difference(out_image_batch, index_image_batch))
        loss = 0.2*loss1 + 0.8*loss2
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss(out_orl)', loss1)
        tf.summary.scalar('loss(out_index)', loss2)
    return loss1, loss2, loss

def losses1123(out_image_batch, index_image_batch):
    with tf.name_scope("LOSS") as scope:
        loss = tf.reduce_mean(tf.squared_difference(out_image_batch, index_image_batch))
        tf.summary.scalar('loss', loss)
        return loss

def losses1124(out_image_batch, index_image_batch):
    with tf.name_scope("LOSS") as scope:
        loss = tf.reduce_mean(tf.squared_difference(out_image_batch, index_image_batch))
        tf.summary.scalar('loss', loss)
        return loss

def losses1126(out_image_batch, index_image_batch, ab_image_batch):
    with tf.name_scope("LOSS") as scope:
        loss1 = tf.reduce_mean(tf.squared_difference(out_image_batch, ab_image_batch))
        loss2 = tf.reduce_mean(tf.squared_difference(out_image_batch, index_image_batch))
        loss = 0.1*loss1 + 0.9*loss2
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss(out_orl)', loss1)
        tf.summary.scalar('loss(out_index)', loss2)
    return loss1, loss2, loss

def losses1127(out_ab_batch, layer1_batch, layer2_batch, index_batch, index_layer1, index_layer2):
    with tf.name_scope("LOSS") as scope:
        loss1 = tf.reduce_mean(tf.squared_difference(layer1_batch, index_layer1))
        loss2 = tf.reduce_mean(tf.squared_difference(layer2_batch, index_layer2))
        loss3 = tf.reduce_mean(tf.squared_difference(out_ab_batch, index_batch))
        loss = loss1 + loss2 + loss3
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss1', loss1)
        tf.summary.scalar('loss2', loss2)
        tf.summary.scalar('loss3', loss3)
        return loss


# 训练操作
def training(loss, global_step):
    with tf.name_scope('OPTIMIZE') as scope:
        lr = tf.train.exponential_decay(1e-3, global_step, 10000, 0.7, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op
