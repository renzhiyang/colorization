# training.py  训练、测试
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import input_data
import model_themeRecommend as model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 50
CAPACITY = 1000     # 队列容量
MAX_STEP = 15000
IMAGE_SIZE = 224

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU

def get_image(image, image_size, type, channel = 3):
    height = image_size[0]
    width = image_size[1]

    if type == "jpg":
        input_image = tf.image.decode_jpeg(image, channels = channel)
    if type == "bmp":
        input_image = tf.image.decode_bmp(image, channels = channel)
    input_image = tf.image.resize_images(input_image, [height, width])
    input_image = tf.cast(input_image, tf.float32) / 255

    if channel == 3:
        input_gray = tf.image.rgb_to_grayscale(input_image)
    elif channel == 1:
        input_gray = input_image
    else:
        print("input image channel is wrong")

    input_gray = tf.reshape(input_gray, [1, height, width, 1])
    return input_gray

def get_imageSize(image, sess, type):
    if type == "jpg":
        image = tf.image.decode_jpeg(image, channels=3)
    if type == "bmp":
        image = tf.image.decode_bmp(image, channels=3)
    img = sess.run(image)
    image_size = img.shape[0:2]
    height = image_size[0] // 8 * 8
    width = image_size[1] // 8 * 8
    return [height, width]

#use rgb train
def test_rgb_recommend():
    test_dir = "test/test_images2/3.jpg"
    checkpoint_Dir = "logs/themeLogs/themeRecommedn6/model.ckpt-39999"
    out_theme_Dir = "output/themeRecommend/recommend6/3.jpg"

    input_image = tf.read_file(test_dir)
    sess = tf.Session()
    #image_size = get_imageSize(input_image, sess, "jpg")
    input_gray = get_image(input_image, [224, 224], type = "jpg", channel = 3)

    out_21 = model.built_network(input_gray)
    out_theme = tf.reshape(out_21, [1, 1, -1, 3])

    logs_dir = 'F:/Deep Learning/Code/colorization/logs/log_gloabal&local/gradient_index4/'
    saver = tf.train.Saver()
    print('载入检查点...')

    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt.model_checkpoint_path = checkpoint_Dir
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    grayImage, theme = sess.run([input_gray, out_theme])
    grayImage = grayImage[0]
    theme = theme[0]
    print(theme*255)

    plt.subplot(1, 2, 1), plt.imshow(theme)
    #plt.subplot(1, 2, 2), plt.imshow(grayImage[0])
    plt.show()

    plt.imsave(out_theme_Dir, theme)

def get_l_channel(input_image, type, image_size, channel):
    if type == "jpg":
        image = tf.image.decode_jpeg(image, channels=channel)
    if type == "bmp":
        image = tf.image.decode_bmp(image, channels=channel)
    image = tf.image.resize_images(image, image_size)
    image = tf.cast(image, tf.float64)
    if channel == 1:
        image = tf.image.grayscale_to_rgb(image)

    image_lab = tf.cast(input_data.rgb_to_lab(image), tf.float32)
    image_l = tf.reshape(image_lab[:, :, :, 0:1], [1, image_size[0], image_size[1], 1]) / 100
    return image_l


#use lab train
def test_lab_recommend():
    test_dir = "test/test_images2/3.jpg"
    checkpoint_Dir = "logs/themeLogs/themeRecommedn6/model.ckpt-39999"
    out_theme_Dir = "output/themeRecommend/recommend6/3.jpg"
    input_image = tf.read_file(test_dir)

    input_gray = get_l_channel(input_image, "jpg", [224,224], 1)

    out_21 = model.built_network(input_gray)
    out_theme = tf.reshape(out_21, [1, 1, -1, 3])

    logs_dir = 'F:/Deep Learning/Code/colorization/logs/log_gloabal&local/gradient_index4/'
    saver = tf.train.Saver()
    print('载入检查点...')

    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt.model_checkpoint_path = checkpoint_Dir
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    theme_lab = sess.run([out_theme])
    theme_lab = theme_lab[0]
    theme_lab[:, :, :, 0:1] = theme_lab[:, :, :, 0:1] * 100
    theme_lab[:, :, :, 1:] = theme_lab[:, :, :, 1:] * 255 - 128
    theme_rgb = color.lab2rgb(theme_lab)
    plt.imshow(theme_rgb)
    plt.show()
    plt.imsave(out_theme_Dir, theme_rgb)

if __name__ == '__main__':
    test_rgb_recommend()
    test_lab_recommend()