# training.py  训练、测试
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import input_data
import model_globalInput as model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 10
CAPACITY = 1000     # 队列容量
MAX_STEP = 150000
IMAGE_SIZE = 224

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU

#theme recommend system
def run_training():
    train_dir = "G:\\Database\\ColoredData\\new_colorimage1"
    index_dir = ""

    logs_dir = "F:\\Project_Yang\\Code\\mainProject\\logs\\log_global\\image loss 0.2"
    result_dir = "results/global/"

    # 获取输入
    image_list = input_data.get_themeRecommend_list(train_dir, index_dir)
    #train batch[BATCH_SIZE, 224, 224, 3], index batch[BATCH_SIZE, 1, 7, 3]
    train_batch, index_rgb_batch = input_data.get_themeRecommend_batch(image_list, BATCH_SIZE, CAPACITY)

    index_batch = tf.reshape(index_batch, [BATCH_SIZE, 1, -1])
    #out_batch [BATCH_SIZE, 1, 21]
    out_batch = model.new_built_network(train_batch)
    out_rgb_batch = tf.reshape(out_batch, [BATCH_SIZE, -1, 3])
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss = model.whole_loss(out_ab_batch, index_batch)
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

            if step % 2000 == 0:
                train_img, index_theme, out_theme = sess.run([train_batch, index_rgb_batch, out_rgb_batch])
                plt.subplot(1, 3, 1), plt.imshow(train_img)
                plt.subplot(1, 3, 2), plt.imshow(index_theme)
                plt.subplot(1, 3, 3), plt.imshow(out_theme)
                plt.savefig(result_dir + str(step) + "_image.png")
                plt.show()

    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()

#run_training()
#test_one_image()
test_theme_image()
# test_batch_image()
