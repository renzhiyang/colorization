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

#theme recommend system
def run_training():
    train_dir = "D:\\themeProject\\Database\\ColorImages"
    index_dir = "D:\\themeProject\\Database\\ColorTheme7"

    logs_dir = "D:\\themeProject\\logs\\themeRecommend"
    result_dir = "themeResult/"

    # 获取输入
    image_list = input_data.get_themeRecommend_list(train_dir, index_dir)
    #train batch[BATCH_SIZE, 224, 224, 3], index batch[BATCH_SIZE, 1, 7, 3]
    train_rgb_batch, index_rgb_batch = input_data.get_themeRecommend_batch(image_list, BATCH_SIZE, CAPACITY)

    train_lab_batch = tf.cast(input_data.rgb_to_lab(train_rgb_batch), dtype = tf.float32)
    index_lab_batch = tf.cast(input_data.rgb_to_lab(index_rgb_batch), dtype=tf.float32)

    #normalize
    train_l_batch = train_lab_batch[:, :, :, 0:1] / 100
    train_ab_batch = (train_lab_batch[:, :, :, 1:] + 128) / 255
    index_l_batch = index_lab_batch[:, :, :, 0:1] / 100
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255
    train_n_batch = tf.concat([train_l_batch, train_ab_batch], 3)
    index_n_batch = tf.concat([index_l_batch, index_ab_batch], 3)

    index_n_batch = tf.reshape(index_n_batch, [BATCH_SIZE, 1, -1])
    #out_batch [BATCH_SIZE, 1, 21]
    out_batch = model.built_network(train_l_batch)
    print(out_batch)
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss = model.whole_loss(out_batch, index_n_batch)
    train_op = model.training(train_loss, global_step)

    out_lab_batch = tf.cast(tf.reshape(out_batch, [BATCH_SIZE, 1, 7, 3]), tf.float64)
    index_n_batch = tf.cast(tf.reshape(index_n_batch, [BATCH_SIZE, 1, 7, 3]), tf.float64)
    train_n_batch = tf.cast(train_n_batch, tf.float64)
    index_lab_batch = tf.cast(index_lab_batch, tf.float64)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=20)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss = sess.run([train_op, train_loss])
            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,  loss: %g" % (step, tra_loss))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 100 == 0:
                train_lab, index_lab, out_lab, index = sess.run([train_n_batch, index_n_batch, out_lab_batch, index_lab_batch])
                train_lab = train_lab[0]
                index_lab = index_lab[0]
                out_lab = out_lab[0]
                index = index[0]


                train_lab[:,:,0:1] = train_lab[:,:,0:1] * 100
                train_lab[:, :, 1:] = train_lab[:,:,1:] * 255 - 128
                index_lab[:, :, 0:1] = index_lab[:, :, 0:1] * 100
                index_lab[:, :, 1:] = index_lab[:, :, 1:] * 255 - 128
                out_lab[:, :, 0:1] = out_lab[:, :, 0:1] * 100
                out_lab[:, :, 1:] = out_lab[:, :, 1:] * 255 - 128
                print(out_lab)


                train_rgb = color.lab2rgb(train_lab)
                index_rgb = color.lab2rgb(index_lab)
                out_rgb = color.lab2rgb(out_lab)


                plt.subplot(1, 3, 1), plt.imshow(train_rgb)
                plt.subplot(1, 3, 2), plt.imshow(index_rgb)
                plt.subplot(1, 3, 3), plt.imshow(out_rgb)
                plt.savefig(result_dir + str(step) + "_image.png")
                #plt.show()

    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()

#use rgb train
def run_training2():
    train_dir = "D:\\themeProject\\Database\\test"
    index_dir = "D:\\themeProject\\Database\\ColorTheme7"

    logs_dir = "D:\\themeProject\\logs\\themeRecommend2"
    result_dir = "themeResult/themeRecommend2/"

    # 获取输入
    image_list = input_data.get_themeRecommend_list(train_dir, index_dir)
    #train batch[BATCH_SIZE, 224, 224, 3], index batch[BATCH_SIZE, 1, 7, 3]
    train_g_batch, train_rgb_batch, index_rgb_batch = input_data.get_themeRecommend_rgb_batch(image_list, BATCH_SIZE, CAPACITY)

    train_rgb_batch = tf.cast(train_rgb_batch, dtype=tf.float32)
    index_rgb_batch = tf.cast(index_rgb_batch, dtype=tf.float32)
    train_g_batch = tf.cast(train_g_batch, dtype=tf.float32)
    index_batch = tf.reshape(index_rgb_batch, [BATCH_SIZE, 21])

    #out_batch [BATCH_SIZE, 1, 21]
    print("ready to model")
    out_batch = model.built_network(train_g_batch)
    print(out_batch)
    out_rgb_batch = tf.reshape(out_batch, [BATCH_SIZE, 1, 7, 3])
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss = model.whole_loss(out_batch, index_batch)
    train_op = model.training(train_loss, global_step)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    saver = tf.train.Saver(max_to_keep=20)

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss = sess.run([train_op, train_loss])
            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,  loss: %g" % (step, tra_loss))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0:
                train_g, train_rgb, index_rgb, out_rgb = sess.run([train_g_batch, train_rgb_batch, index_rgb_batch, out_rgb_batch])
                train_rgb = train_rgb[0]
                index_rgb = index_rgb[0]
                out_rgb = out_rgb[0]
                train_g = train_g[0]

                print(out_rgb)
                print(index_rgb)

                plt.subplot(1, 4, 1), plt.imshow(train_rgb)
                #plt.subplot(1, 4, 2), plt.imshow(train_g)
                plt.subplot(1, 4, 2), plt.imshow(out_rgb)
                plt.subplot(1, 4, 3), plt.imshow(index_rgb)
                plt.savefig(result_dir + str(step) + "_image.png")
                #plt.show()

    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()

run_training2()
#test_one_image()
#test_theme_image()
# test_batch_image()