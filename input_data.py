# input_data.py  获取数据

import tensorflow as tf
import numpy as np
import os
import skimage.color as color
from matplotlib import pyplot as plt


# RGB空间转LAB空间，输入的RGB已经除以255归一化，输出的LAB未归一化
def rgb_to_lab(image):
    assert image.get_shape()[-1] == 3

    rgb_pixels = tf.reshape(image, [-1, 3])
    # RGB转XYZ
    with tf.name_scope("rgb_to_xyz"):
        linear_mask = tf.cast(rgb_pixels <= 0.04045, dtype=tf.float64)
        expoential_mask = tf.cast(rgb_pixels > 0.04045, dtype=tf.float64)
        rgb_pixels = (rgb_pixels / 12.92) * linear_mask +\
                     (((rgb_pixels + 0.055) / 1.055) ** 2.4) * expoential_mask
        transfer_mat = tf.constant([
            [0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227]
        ], dtype=tf.float64)
        xyz_pixels = tf.matmul(rgb_pixels, transfer_mat)

    # XYZ转LAB
    with tf.name_scope("xyz_to_lab"):
        # 标准化D65白点
        xyz_norm_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
        # xyz_norm_pixels = tf.multiply(xyz_pixels, [1 / 0.95047, 1.0, 1 / 1.08883])
        epsilon = 6/29
        linear_mask = tf.cast(xyz_norm_pixels <= epsilon**3, dtype=tf.float64)
        expoential_mask = tf.cast(xyz_norm_pixels > epsilon**3, dtype=tf.float64)
        f_xyf_pixels = (xyz_norm_pixels / (3 * epsilon**2) + 4/29) * linear_mask +\
                       (xyz_norm_pixels**(1/3)) * expoential_mask
        transfer_mat2 = tf.constant([
            [0.0, 500.0, 0.0],
            [116.0, -500.0, 200.0],
            [0.0, 0.0, -200.0]
        ], dtype=tf.float64)
        lab_pixels = tf.matmul(f_xyf_pixels, transfer_mat2) + tf.constant([-16.0, 0.0, 0.0], dtype=tf.float64)

        image_lab = tf.reshape(lab_pixels, tf.shape(image))

    return image_lab

# 递归遍历所有文件
def get_all_files(file_path):
    filename_list = []
    for item in os.listdir(file_path):
        path = file_path + '\\' + item
        if os.path.isdir(path):     # 如果是文件夹
            filename_list.extend(get_all_files(path))
        elif os.path.isfile(path):  # 如果是文件
            filename_list.append(path)
        #if len(filename_list) > 10000:
        #   break
    return filename_list


# 获取指定路径下的训练数据和真值
def get_image_list(train_dir, sparse_dir, mask_dir, index_dir):
    train_list = get_all_files(train_dir)       # 彩色图片
    #train_list.extend(train_list)
    #train_list.extend(train_list)
    sparse_list = get_all_files(sparse_dir)       # 颜色主题
    index_list = get_all_files(index_dir)       # 颜色标签
    mask_list = get_all_files(mask_dir)

    print("训练目录%s, 文件个数%d" % (train_dir, len(train_list)))
    print("训练目录%s, 文件个数%d" % (sparse_dir, len(sparse_list)))
    print("训练目录%s, 文件个数%d" % (index_dir, len(index_list)))

    temp = np.array([train_list, sparse_list, mask_list, index_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    train_list = list(temp[:, 0])
    sparse_list = list(temp[:, 1])
    mask_list = list(temp[:, 2])
    index_list = list(temp[:, 3])

    return [train_list, sparse_list, mask_list, index_list]


# 按批次获取图片
def get_batch(file_list, batch_size, capacity):
    image_size = 224

    # 生成队列
    filename_queue = tf.train.slice_input_producer(file_list, shuffle=False)

    # 彩色图片处理
    train_image = tf.read_file(filename_queue[0])
    train_image = tf.image.decode_jpeg(train_image, channels=3)
    #train_image = tf.image.decode_bmp(train_image)
    train_image = tf.image.resize_images(train_image, [image_size, image_size])
    train_image = tf.cast(train_image, tf.float64) / 255.0     # 转LAB空间需要float64
    train_image = rgb_to_lab(train_image)
    train_image = tf.cast(train_image, tf.float32)      # 神经网络需要float32
    l_color = train_image[:, :, 0] / 100.0       # l范围[0, 100]
    l_color = tf.reshape(l_color, [image_size, image_size, 1])
    ab_color = (train_image[:, :, 1:] + 128) / 255.0     # ab范围[-128, 127]
    train_image = tf.concat([l_color, ab_color], 2)

    train_theme = tf.read_file(filename_queue[1])
    #train_theme = tf.image.decode_jpeg(train_theme, channels=3)
    train_theme = tf.image.decode_bmp(train_theme, channels = 3)
    train_theme = tf.image.resize_images(train_theme, [image_size, image_size])
    train_theme = tf.cast(train_theme, tf.float64) / 255.0  # 转LAB空间需要float64
    train_theme = rgb_to_lab(train_theme)
    train_theme = tf.cast(train_theme, tf.float32)  # 神经网络需要float32
    l_theme = train_theme[:, :, 0] / 100.0  # l范围[0, 100]
    l_theme = tf.reshape(l_theme, [image_size, image_size, 1])
    ab_theme = (train_theme[:, :, 1:] + 128) / 255.0  # ab范围[-128, 127]
    sparse = tf.concat([l_theme, ab_theme], 2)

    #mask_batch
    train_mask = tf.read_file(filename_queue[2])
    train_mask = tf.image.decode_bmp(train_mask, channels = 3)
    train_mask = tf.image.resize_images(train_mask, [image_size, image_size])
    train_mask = tf.cast(train_mask, tf.float32) / 255.0
    train_mask = train_mask[:, :, :1]


    # 颜色标签处理
    train_index = tf.read_file(filename_queue[3])
    train_index = tf.image.decode_jpeg(train_index, channels=3)
    train_index = tf.image.resize_images(train_index, [image_size, image_size])
    train_index = tf.cast(train_index, tf.float64) / 255.0
    train_index = rgb_to_lab(train_index)
    train_index = tf.cast(train_index, tf.float32)
    # l_index = train_index[:, :, 0] / 100.0
    # l_index = tf.reshape(l_index, [image_size, image_size, 1])
    ab_index = (train_index[:, :, 1:] + 128) / 255.0

    # 获取batch
    l_batch, ab_bacth, lab_batch, sparse_ab_batch, index_batch, mask_batch=\
        tf.train.shuffle_batch([l_color, ab_color, train_image, ab_theme, ab_index, train_mask],
                               batch_size=batch_size,
                               capacity=capacity,
                               min_after_dequeue=500,
                               num_threads=64)

    return l_batch, ab_bacth, lab_batch, sparse_ab_batch, index_batch, mask_batch

def get_image_list2(train_dir, mask_dir, index_dir):
    train_list = get_all_files(train_dir)       # 彩色图片
    #train_list.extend(train_list)
    #train_list.extend(train_list)
    #sparse_list = get_all_files(sparse_dir)       # 颜色主题
    index_list = get_all_files(index_dir)       # 颜色标签
    mask_list = get_all_files(mask_dir)

    print("训练目录%s, 文件个数%d" % (train_dir, len(train_list)))
    print("训练目录%s, 文件个数%d" % (mask_dir, len(mask_list)))
    print("训练目录%s, 文件个数%d" % (index_dir, len(index_list)))

    temp = np.array([train_list, mask_list, index_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    train_list = list(temp[:, 0])
    #sparse_list = list(temp[:, 1])
    mask_list = list(temp[:, 1])
    index_list = list(temp[:, 2])

    return [train_list, mask_list, index_list]

def get_batch2(file_list, batch_size, capacity):
    image_size = 224

    # 生成队列
    filename_queue = tf.train.slice_input_producer(file_list, shuffle=False)

    # 彩色图片处理
    train_image = tf.read_file(filename_queue[0])
    train_image = tf.image.decode_jpeg(train_image, channels=3)
    #train_image = tf.image.decode_bmp(train_image)
    train_image = tf.image.resize_images(train_image, [image_size, image_size])
    train_image = tf.cast(train_image, tf.float64) / 255.0     # 转LAB空间需要float64
    train_image = rgb_to_lab(train_image)
    train_image = tf.cast(train_image, tf.float32)      # 神经网络需要float32
    #l_color = train_image[:, :, 0] / 100.0       # l范围[0, 100]
    l_color = train_image[:, :, 0]
    l_color = tf.reshape(l_color, [image_size, image_size, 1])
    #ab_color = (train_image[:, :, 1:] + 128) / 255.0     # ab范围[-128, 127]
    ab_color = train_image[:, :, 1:]
    train_image = tf.concat([l_color, ab_color], 2)

    #mask_batch
    train_mask = tf.read_file(filename_queue[1])
    train_mask = tf.image.decode_bmp(train_mask, channels = 3)
    train_mask = tf.image.resize_images(train_mask, [image_size, image_size])
    train_mask = tf.cast(train_mask, tf.float32) / 255.0
    train_mask_2channels = train_mask[:, :, 0:2]
    train_mask = train_mask[:, :, 0]
    train_mask = tf.reshape(train_mask, [image_size, image_size, 1])


    # 颜色标签处理
    train_index = tf.read_file(filename_queue[2])
    train_index = tf.image.decode_jpeg(train_index, channels=3)
    train_index = tf.image.resize_images(train_index, [image_size, image_size])
    train_index = tf.cast(train_index, tf.float64) / 255.0
    train_index = rgb_to_lab(train_index)
    train_index = tf.cast(train_index, tf.float32)
    # l_index = train_index[:, :, 0] / 100.0
    # l_index = tf.reshape(l_index, [image_size, image_size, 1])
    #ab_index = (train_index[:, :, 1:] + 128) / 255.0
    ab_index = train_index[:, :, 1:]

    # 获取batch
    l_batch, ab_bacth, index_batch, mask_batch, mask_batch_2channels =\
        tf.train.shuffle_batch([l_color, ab_color, ab_index, train_mask, train_mask_2channels],
                               batch_size=batch_size,
                               capacity=capacity,
                               min_after_dequeue=500,
                               num_threads=64)

    return l_batch, ab_bacth, index_batch, mask_batch, mask_batch_2channels

def get_image_list1219(train_dir, mask_dir):
    train_list = get_all_files(train_dir)
    mask_list = get_all_files(mask_dir)
    print("训练目录%s, 文件个数%d" % (train_dir, len(train_list)))
    print("训练目录%s, 文件个数%d" % (mask_dir, len(mask_list)))

    temp = np.array([train_list, mask_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    train_list = list(temp[:, 0])
    #sparse_list = list(temp[:, 1])
    mask_list = list(temp[:, 1])

    return [train_list, mask_list]


def get_batch_1219(file_list, batch_size, capacity):
    image_size = 224

    # 生成队列
    filename_queue = tf.train.slice_input_producer(file_list, shuffle=False)

    # 彩色图片处理
    train_image = tf.read_file(filename_queue[0])
    print(filename_queue[0])
    train_image = tf.image.decode_jpeg(train_image, channels=3)
    print(train_image)
    #train_image = tf.image.decode_bmp(train_image)
    train_image = tf.image.resize_images(train_image, [image_size, image_size])
    train_image = tf.cast(train_image, tf.float64) / 255.0     # 转LAB空间需要float64
    train_image = rgb_to_lab(train_image)
    train_image = tf.cast(train_image, tf.float32)      # 神经网络需要float32
    #l_color = train_image[:, :, 0] / 100.0       # l范围[0, 100]
    l_color = train_image[:, :, 0]
    l_color = tf.reshape(l_color, [image_size, image_size, 1])
    #ab_color = (train_image[:, :, 1:] + 128) / 255.0     # ab范围[-128, 127]
    ab_color = train_image[:, :, 1:]
    train_image = tf.concat([l_color, ab_color], 2)

    #mask_batch
    train_mask = tf.read_file(filename_queue[1])
    print(filename_queue[1])
    train_mask = tf.image.decode_bmp(train_mask, channels = 3)
    print(train_mask)
    train_mask = tf.image.resize_images(train_mask, [image_size, image_size])
    train_mask = tf.cast(train_mask, tf.float32) / 255.0
    train_mask_2channels = train_mask[:, :, 0:2]
    train_mask = train_mask[:, :, 0]
    train_mask = tf.reshape(train_mask, [image_size, image_size, 1])
    # 获取batch
    l_batch, ab_bacth, mask_batch, mask_batch_2channels =\
        tf.train.shuffle_batch([l_color, ab_color, train_mask, train_mask_2channels],
                               batch_size=batch_size,
                               capacity=capacity,
                               min_after_dequeue=500,
                               num_threads=64)

    return l_batch, ab_bacth, mask_batch, mask_batch_2channels


def get_themeInput_list(train_dir, theme_dir, theme_index_dir, image_index_dir):
    train_list = get_all_files(train_dir)
    theme_list = get_all_files(theme_dir)
    theme_index_list = get_all_files(theme_index_dir)
    image_index_list = get_all_files(image_index_dir)

    print("训练目录%s, 文件个数%d" % (train_dir, len(train_list)))
    print("训练目录%s, 文件个数%d" % (theme_dir, len(theme_list)))
    print("训练目录%s, 文件个数%d" % (theme_index_dir, len(theme_index_list)))
    print("训练目录%s, 文件个数%d" % (image_index_dir, len(image_index_list)))

    temp = np.array([train_list, theme_list, theme_index_list, image_index_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    train_list = list(temp[:, 0])
    theme_list = list(temp[:, 1])
    theme_index_list = list(temp[:, 2])
    image_index_list = list(temp[:, 3])

    return [train_list, theme_list, theme_index_list, image_index_list]

def get_themeObj_batch(file_list, batch_size, capacity):
    image_size = 224

    # produce queue
    filename_queue = tf.train.slice_input_producer(file_list, shuffle=False)

    # train image list
    train_image = tf.read_file(filename_queue[0])
    train_image = tf.image.decode_jpeg(train_image, channels=3)
    train_image = tf.image.resize_images(train_image, [image_size, image_size])
    train_image = tf.cast(train_image, tf.float32) / 255


    #theme list
    theme = tf.read_file(filename_queue[1])
    theme = tf.image.decode_bmp(theme, channels=3)
    theme = tf.image.resize_images(theme, [1, 5])
    theme = tf.cast(theme, tf.float32) / 255

    #theme index list
    theme_index = tf.read_file(filename_queue[2])
    theme_index = tf.image.decode_jpeg(theme_index, channels=3)
    theme_index = tf.image.resize_images(theme_index, [image_size, image_size])
    theme_index = tf.cast(theme_index, tf.float32) / 255

    #theme mask list
    '''theme_mask = tf.read_file(filename_queue[3])
    theme_mask = tf.image.decode_bmp(theme_mask, channels=3)
    theme_mask = tf.image.resize_images(theme_mask, [1, 5])
    theme_mask = tf.cast(theme_mask[:, :, 0], tf.float32) / 255
    theme_mask = tf.reshape(theme_mask, [1, 5, 1])'''
    theme_mask = tf.ones([1, 5, 1], dtype = tf.float32)

    #image index list
    image_index = tf.read_file(filename_queue[3])
    image_index = tf.image.decode_jpeg(image_index, channels=3)
    image_index = tf.image.resize_images(image_index, [image_size, image_size])
    image_index = tf.cast(image_index, tf.float32) / 255

    train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch = \
        tf.train.shuffle_batch([train_image, theme, theme_index, theme_mask, image_index],
                               batch_size=batch_size,
                               capacity=capacity,
                               min_after_dequeue=500,
                               num_threads=64)

    return train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch

def get_wholeInput_list(train_dir, theme_dir, theme_index_dir, image_index_dir, sparse_mask_dir):
    train_list = get_all_files(train_dir)
    theme_list = get_all_files(theme_dir)
    theme_index_list = get_all_files(theme_index_dir)
    image_index_list = get_all_files(image_index_dir)
    sparse_mask_list = get_all_files(sparse_mask_dir)

    print("训练目录%s, 文件个数%d" % (train_dir, len(train_list)))
    print("训练目录%s, 文件个数%d" % (theme_dir, len(theme_list)))
    print("训练目录%s, 文件个数%d" % (theme_index_dir, len(theme_index_list)))
    print("训练目录%s, 文件个数%d" % (image_index_dir, len(image_index_list)))
    print("训练目录%s, 文件个数%d" % (sparse_mask_dir, len(sparse_mask_list)))

    temp = np.array([train_list, theme_list, theme_index_list, image_index_list, sparse_mask_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    train_list = list(temp[:, 0])
    theme_list = list(temp[:, 1])
    theme_index_list = list(temp[:, 2])
    image_index_list = list(temp[:, 3])
    sparse_mask_list = list(temp[:, 4])

    return [train_list, theme_list, theme_index_list, image_index_list, sparse_mask_list]

def get_wholeObj_batch(file_list, batch_size, capacity):
    image_size = 224

    # produce queue
    filename_queue = tf.train.slice_input_producer(file_list, shuffle=False)

    # train image list
    train_image = tf.read_file(filename_queue[0])
    train_image = tf.image.decode_jpeg(train_image, channels=3)
    train_image = tf.image.resize_images(train_image, [image_size, image_size])
    train_image = tf.cast(train_image, tf.float32) / 255


    #theme list
    theme = tf.read_file(filename_queue[1])
    theme = tf.image.decode_bmp(theme, channels=3)
    theme = tf.image.resize_images(theme, [1, 5])
    theme = tf.cast(theme, tf.float32) / 255

    #theme index list
    theme_index = tf.read_file(filename_queue[2])
    theme_index = tf.image.decode_jpeg(theme_index, channels=3)
    theme_index = tf.image.resize_images(theme_index, [image_size, image_size])
    theme_index = tf.cast(theme_index, tf.float32) / 255

    #theme mask list
    theme_mask = tf.ones([1, 5, 1], dtype = tf.float32)

    #image index list
    image_index = tf.read_file(filename_queue[3])
    image_index = tf.image.decode_jpeg(image_index, channels=3)
    image_index = tf.image.resize_images(image_index, [image_size, image_size])
    image_index = tf.cast(image_index, tf.float32) / 255

    #spare mask list
    sparse_mask = tf.read_file(filename_queue[4])
    sparse_mask = tf.image.decode_bmp(sparse_mask, channels=3)
    sparse_mask = tf.image.resize_images(sparse_mask, [image_size, image_size])
    sparse_mask = tf.cast(sparse_mask, tf.float32) / 255
    sparse_mask2channels = sparse_mask[:, :, 0:2]


    train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch, sparse_mask2channels_batch = \
        tf.train.shuffle_batch([train_image, theme, theme_index, theme_mask, image_index, sparse_mask2channels],
                               batch_size=batch_size,
                               capacity=capacity,
                               min_after_dequeue=500,
                               num_threads=64)

    return train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch, sparse_mask2channels_batch


