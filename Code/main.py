# training.py  训练、测试

import tensorflow as tf
import numpy as np
import os
import input_data
import model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 10
CAPACITY = 1000     # 队列容量
MAX_STEP = 100000

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU


def run_training1():
    #train_dir = "F:\\Project_Yang\\Database\\new_colorimage1"
    #sparse_dir = "F:\\Project_Yang\\Code\\Project_Yang\\sparseimg"
    #index_dir = "F:\\Project_Yang\\Database\\new_colorimage4"
    train_dir = "F:\\Project_Yang\\Database\\training_image1000"
    sparse_dir = "F:\\Project_Yang\\Database\\sparse_image4_1000bmp"
    index_dir = "F:\\Project_Yang\\Database\\index_image1000"
    logs_dir = "log1126\\"

    # 获取输入
    image_list = input_data.get_image_list(train_dir, sparse_dir, index_dir)
    l_batch, ab_batch, lab_batch, sparse_ab_batch, index_batch = input_data.get_batch(image_list, BATCH_SIZE, CAPACITY)

    out_ab_batch = model.built_network1126(lab_batch, sparse_ab_batch)

    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    loss1, loss2, train_loss = model.losses1126(out_ab_batch, index_batch, ab_batch)
    train_op = model.training(train_loss, global_step)

    l_batch = tf.cast(l_batch, tf.float64)
    lab_batch = tf.cast(lab_batch, tf.float64)

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

            _, tra_loss, los1, los2 = sess.run([train_op, train_loss, loss1, loss2])

            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d, loss(out_orl): %g, loss(out_index): %g, loss: %g" % (step, los1, los2, tra_loss))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            '''if step - 1e4 == 0:
                l, lab, ab_out = sess.run([l_batch, lab_batch, out_ab_batch])
                l = l[0]
                lab = lab[0]
                ab_out = ab_out[0]

                print([l[:, :, 0].min(), l[:, :, 0].max()])
                print([ab_out[:, :, 0].min(), ab_out[:, :, 0].max()])
                print([ab_out[:, :, 1].min(), ab_out[:, :, 1].max()])

                l = l * 100
                lab = lab * 255 - 128
                ab_out = ab_out * 255 - 128
                img_in = np.concatenate([l, lab], 2)
                img_in = color.lab2rgb(img_in)
                img_out = np.concatenate([l, ab_out], 2)
                img_out = color.lab2rgb(img_out)

                print([l[:, :, 0].min(), l[:, :, 0].max()])
                print([ab_out[:, :, 0].min(), ab_out[:, :, 0].max()])
                print([ab_out[:, :, 1].min(), ab_out[:, :, 1].max()])
                print()
                plt.subplot(241), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(242), plt.imshow(lab[:, :, 0], 'gray')
                plt.subplot(243), plt.imshow(lab[:, :, 1], 'gray')
                plt.subplot(244), plt.imshow(img_in)
                plt.subplot(245), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(246), plt.imshow(ab_out[:, :, 0], 'gray')
                plt.subplot(247), plt.imshow(ab_out[:, :, 1], 'gray')
                plt.subplot(248), plt.imshow(img_out)
                plt.show()
                '''

    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()


def get_l_channel(image, image_size):
    l_channel = tf.image.decode_jpeg(image, channels=3)
    l_channel = tf.image.resize_images(l_channel, [image_size, image_size])
    l_channel = tf.cast(l_channel, tf.float64) / 255.0
    l_channel = input_data.rgb_to_lab(l_channel)
    l_channel = tf.cast(l_channel, tf.float32)
    ab_channel = (l_channel[:, :, 1:] + 128) / 255.0
    ab_channel = tf.reshape(ab_channel, [1, image_size, image_size, 2])
    l_channel = l_channel[:, :, 0] / 100.0
    l_channel = tf.reshape(l_channel, [1, image_size, image_size, 1])
    # l_channel = tf.cast(l_channel, tf.float64)

    return l_channel, ab_channel


def get_theme(train_theme):
    train_theme = tf.decode_raw(train_theme, tf.uint8)
    # train_theme = tf.reshape(train_theme, [70]); train_theme = train_theme[-16:-1]
    train_theme = tf.reshape(train_theme, [72]); train_theme = train_theme[-18:-3]
    train_theme = tf.reshape(train_theme, [1, 5, 3])
    R = train_theme[:, :, 2]
    G = train_theme[:, :, 1]
    B = train_theme[:, :, 0]
    R = tf.reshape(R, [1, 5, 1])
    G = tf.reshape(G, [1, 5, 1])
    B = tf.reshape(B, [1, 5, 1])
    train_theme = tf.concat([R, G, B], 2)
    train_theme = tf.cast(train_theme, tf.float64) / 255.0
    train_theme = input_data.rgb_to_lab(train_theme)
    train_theme = tf.cast(train_theme, tf.float32)
    ab_theme = (train_theme[:, :, 1:] + 128) / 255.0

    return ab_theme


def get_output(l, ab):
    l = l * 100.0
    ab = ab * 255.0 - 128
    img_out = np.concatenate([l, ab], 2)
    img_out = color.lab2rgb(img_out)

    return img_out


def eval_one_image():
    image_size = 224
    test_image1 = tf.read_file('test_images2/1_2.jpg')
    l_channel, ab_channel = get_l_channel(test_image1, image_size)
    test_input = tf.concat([l_channel, ab_channel], 3)
    # test_image2 = tf.read_file('test_images2/1_2.jpg')
    # l_channel, ab_channel = get_l_channel(test_image2, image_size)
    # test_image3 = tf.read_file('test_images2/1_3.jpg')
    # l_channel, ab_channel = get_l_channel(test_image3, image_size)
    # test_image4 = tf.read_file('test_images2/1_4.jpg')
    # l_channel, ab_channel = get_l_channel(test_image4, image_size)
    # test_image5 = tf.read_file('test_images2/1_5.jpg')
    # l_channel, ab_channel = get_l_channel(test_image5, image_size)
    # test_image6 = tf.read_file('test_images2/1_6.jpg')
    # l_channel, ab_channel = get_l_channel(test_image6, image_size)

    fix = '10'
    theme = tf.read_file('test_images2/theme_' + fix + '.bmp')
    theme = get_theme(theme)

    # TODO: 选择测试模型
    out_ab = model.builtNetwork(test_input, theme)

    # LAB转RGB需要float64
    out_ab = tf.cast(out_ab, tf.float64)

    # TODO: 选择检查点
    logs_dir = 'logs_3_1'
    saver = tf.train.Saver()

    sess = tf.Session()
    print('载入检查点...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    l, ab = sess.run([l_channel, out_ab])
    img_out1 = get_output(l[0], ab[0])
    # img_out2 = get_output(l[1], ab[1])
    # img_out3 = get_output(l[2], ab[2])
    # img_out4 = get_output(l[3], ab[3])
    # img_out5 = get_output(l[4], ab[4])
    # img_out6 = get_output(l[5], ab[5])

    print([l[0, :, :, 0].min(), l[0, :, :, 0].max()])
    print([ab[0, :, :, 0].min(), ab[0, :, :, 0].max()])
    print([ab[0, :, :, 1].min(), ab[0, :, :, 1].max()])
    print()

    plt.imsave('out_images/1_2_testout_' + fix + '.bmp', img_out1)
    # plt.imsave('out_images\\2_testout_' + fix + '.bmp', img_out2)
    # plt.imsave('out_images\\3_testout_' + fix + '.bmp', img_out3)
    # plt.imsave('out_images\\4_testout_' + fix + '.bmp', img_out4)
    # plt.imsave('out_images\\5_testout_' + fix + '.bmp', img_out5)
    # plt.imsave('out_images\\6_testout_' + fix + '.bmp', img_out6)


run_training1()
#eval_one_image()
# test_batch_image()
