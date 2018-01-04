# training.py  训练、测试
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import input_data
import model_wholeInput as model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 10
CAPACITY = 1000     # 队列容量
MAX_STEP = 150000
IMAGE_SIZE = 224

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU

#global theme input, use 3 resnet, 5 color theme
def run_training():
    train_dir = "G:\\Database\\ColoredData\\new_colorimage1"
    theme_index_dir = "G:\\Database\\ColoredData\\ColorMap5_image4"
    image_index_dir = "G:\\Database\\ColoredData\\new_colorimage4"
    theme_dir = "G:\\Database\\ColoredData\\colorImages4_5theme"
    sparse_mask_dir = "F:\\Project_Yang\\Database\\database_new\\mask_image"

    logs_dir = "F:\\Project_Yang\\Code\\mainProject\\logs\\log1814"
    result_dir = "results/1814/"

    # 获取输入
    image_list = input_data.get_wholeInput_list(train_dir, theme_dir, theme_index_dir, image_index_dir, sparse_mask_dir)
    train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch, sparse_mask2channels_batch = input_data.get_wholeObj_batch(image_list, BATCH_SIZE, CAPACITY)

    #rgb_to_lab

    train_batch = tf.cast(train_batch, tf.float64)
    image_index_batch = tf.cast(train_batch, tf.float64)
    theme_batch = tf.cast(theme_batch, tf.float64)
    theme_index_batch = tf.cast(theme_index_batch, tf.float64)
    train_lab_batch = tf.cast(input_data.rgb_to_lab(train_batch), tf.float32)
    theme_lab_batch = tf.cast(input_data.rgb_to_lab(theme_batch), tf.float32)
    index_lab_batch = tf.cast(input_data.rgb_to_lab(image_index_batch), tf.float32)
    themeIndex_lab_batch = tf.cast(input_data.rgb_to_lab(theme_index_batch), tf.float32)

    #do + - * / before normalization
    sparse_ab_batch = sparse_mask2channels_batch * index_ab_batch


    #normalization
    image_l_batch = tf.reshape(train_lab_batch[:, :, :, 0] / 100, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_batch = tf.cast(image_l_batch, tf.float64)
    image_ab_batch = (train_lab_batch[:, :, :, 1:] + 128) / 255
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128) / 255
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255
    sparse_ab_batch = (sparse_ab_batch[:, :, :, 1:] + 128) / 255
    themeIndex_ab_batch = (themeIndex_lab_batch[:, :, :, 1:] + 128) / 255

    #input batches
    theme_input = tf.concat([theme_ab_batch, theme_mask_batch], 3)
    sparse_input = tf.concat([sparse_ab_batch, sparse_mask2channels_batch[:, :, :, 0:1]], 3)

    #concat image_ab and sparse_ab as input
    out_ab_batch = model.built_network(image_ab_batch, theme_input, sparse_input)

    #loss batches
    image_exceptPoints = image_ab_batch - image_ab_batch * sparse_mask2channels_batch
    out_exceptPoints = out_ab_batch - out_ab_batch * sparse_mask2channels_batch

    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    whole_loss, global_loss, local_loss, index_loss, image_loss, color_loss, sobel_loss, exceptPoints_loss = model.whole_loss(
        out_ab_batch, index_ab_batch, themeIndex_ab_batch, image_ab_batch, image_exceptPoints, out_exceptPoints)

    train_rmse, train_psnr = model.get_PSNR(out_ab_batch, index_ab_batch)
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

            _, whole_l, global_l, local_l, index_l, image_l, color_l, sobel_l, exceptPoints_l = sess.run(
                [train_op, whole_loss, global_loss, local_loss, index_loss, image_loss, color_loss, sobel_loss, exceptPoints_loss])
            tra_rmse, tra_psnr = sess.run([train_rmse, train_psnr])

            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,  whole_loss: %g,  global_loss: %g,  local_loss: %g,  index_loss: %g,  image_loss: %g,  color_loss: %g,  sobel_loss: %g,  except_loss: %g  rmse: %g,  psnr: %g"
                      % (step, whole_loss, global_loss, local_loss, index_loss, image_loss, color_loss, sobel_loss, exceptPoints_loss, tra_rmse, tra_rmse))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 2000 == 0:
                l, ab, ab_index, ab_out, theme = sess.run(
                    [image_l_batch, image_ab_batch, index_ab_batch, out_ab_batch, theme_batch])
                l = l[0] * 100
                ab = ab[0] * 255 - 128
                ab_index = ab_index[0] * 255 - 128
                ab_out = ab_out[0] * 255 - 128
                theme = theme[0] * 255

                img_in = np.concatenate([l, ab], 2)
                img_in = color.lab2rgb(img_in)
                img_out = np.concatenate([l, ab_out], 2)
                img_out = color.lab2rgb(img_out)
                img_index = np.concatenate([l, ab_index], 2)
                img_index = color.lab2rgb(img_index)

                plt.subplot(4, 4, 1), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 2), plt.imshow(ab[:, :, 0], 'gray')
                plt.subplot(4, 4, 3), plt.imshow(ab[:, :, 1], 'gray')
                plt.subplot(4, 4, 4), plt.imshow(img_in)

                plt.subplot(4, 4, 5), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 6), plt.imshow(ab_out[:, :, 0], 'gray')
                plt.subplot(4, 4, 7), plt.imshow(ab_out[:, :, 1], 'gray')
                plt.subplot(4, 4, 8), plt.imshow(img_out)

                plt.subplot(4, 4, 9), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 10), plt.imshow(ab_index[:, :, 0], 'gray')
                plt.subplot(4, 4, 11), plt.imshow(ab_index[:, :, 1], 'gray')
                plt.subplot(4, 4, 12), plt.imshow(img_index)

                plt.subplot(4, 4, 13), plt.imshow(theme)
                plt.savefig(result_dir + str(step) + "_image.png")
                plt.show()



                plt.figure(figsize=(8,8))
                axes1 = plt.subplot(221)
                axes1.scatter(ab[:, :, 0], ab[:, :, 1],alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('input images')

                axes2 = plt.subplot(222)
                axes2.scatter(ab_out[:, :, 0], ab_out[:, :, 1],alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('output images')

                axes3 = plt.subplot(223)
                axes3.scatter(ab_index[:, :, 0], ab_index[:, :, 1], alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('index images')

                axes4 = plt.subplot(224)
                part1 = axes4.scatter(ab[:, :, 0], ab[:, :, 1], alpha=0.5, edgecolor='white', label='image_in', s=8)
                part2 = axes4.scatter(ab_out[:, :, 0], ab_out[:, :, 1], alpha=0.5, edgecolor='white', label='image_out', c = 'r', s=8)
                part3 = axes4.scatter(ab_index[:, :, 0], ab_index[:, :, 1], alpha=0.5, edgecolor='white', label='image_index', c='g', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                axes4.legend((part1, part2, part3), ('input', 'output', 'index'))
                plt.savefig(result_dir + str(step) + "_scatter.png")
                plt.show()




    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()

#产生的ab通道是未归一态的
def get_lab_channel(image, image_size, type):
    if type == "jpg":
        l_channel = tf.image.decode_jpeg(image, channels = 3)
    if type == "bmp":
        l_channel = tf.image.decode_bmp(image, channels = 3)

    l_channel = tf.image.resize_images(l_channel, [image_size, image_size])
    l_channel = tf.cast(l_channel, tf.float64) / 255.0
    l_channel = input_data.rgb_to_lab(l_channel)
    l_channel = tf.cast(l_channel, tf.float32)
    ab_channel = l_channel[:, :, 1:]
    #ab_channel = (l_channel[:, :, 1:] + 128) / 255.0
    l_channel = l_channel[:, :, 0] / 100.0
    l_channel = tf.reshape(l_channel, [image_size, image_size, 1])
    ab_channel = tf.reshape(ab_channel, [1, image_size, image_size, 2])
    l_channel = tf.reshape(l_channel, [1, image_size, image_size, 1])
    #lab_channel = tf.reshape(lab_channel, [1, image_size, image_size, 3])
    return l_channel, ab_channel


def get_output(l, ab):
    l = l * 100.0
    ab = ab * 255.0 - 128
    img_out = np.concatenate([l, ab], 2)
    img_out = color.lab2rgb(img_out)

    return img_out

#produce a mask image and save it in mask_dir
def get_mask(sparse_dir, mask_dir, image_size):
    sparse = Image.open(sparse_dir)
    sparse_img = sparse.load()

    maskImg = Image.new("RGB", (image_size, image_size))
    pmaskImg = maskImg.load()

    for i in range(image_size):
        for j in range(image_size):
            if sparse_img[i, j] != (0, 0, 0):
                pmaskImg[i, j] = (255, 255, 255, 255)
    maskImg.save(mask_dir)

def get_mask_channels(mask_img, image_size):
    mask_img = tf.image.decode_bmp(mask_img)
    mask_img = tf.cast(mask_img, tf.float32) / 255.0
    mask_img =  tf.reshape(mask_img, [1, image_size, image_size, 3])
    mask_one_channel = tf.reshape(mask_img[:, :, :, 0], [1, image_size, image_size, 1])
    mask_two_channels = mask_img[:, :, :, 1:]
    return mask_one_channel, mask_two_channels

def get_theme_channels(theme_img, type):
    if type == "jpg":
        theme_img = tf.image.decode_jpeg(theme_img, channels = 3)
    if type == "bmp":
        theme_img = tf.image.decode_bmp(theme_img, channels = 3)
    theme_img = tf.image.resize_images(theme_img, [1, 5])
    theme_img = tf.cast(theme_img, tf.float32) / 255.0
    theme_img = tf.reshape(theme_img, [1, 1, 5, 3])
    theme_ab = theme_img[:, :, :, 1:]
    return theme_ab

def get_theme_mask(theme_mask, type):
    '''if type == "jpg":
        theme_mask = tf.image.decode_jpeg(theme_mask, channels = 3)
    if type == "bmp":
        theme_mask = tf.image.decode_bmp(theme_mask, channels = 3)
    theme_mask = tf.image.resize_images(theme_mask, [1, 7])
    theme_mask = tf.cast(theme_mask, tf.float32) / 255.0
    theme_mask = tf.reshape(theme_mask, [1, 1, 7, 3])
    theme_mask = tf.reshape(theme_mask[:, :, :, 0], [1, 1, 7, 1])'''

    theme_mask = tf.ones([1, 1, 5, 1], dtype=tf.float32)
    return theme_mask

def test_theme_image():
    # sparse_name: blueLine, none, red&blue, red&blue2, redLine
    test_Dir = "test/test_images/27.jpg"
    theme_Dir = "test/test_theme/1 (4).bmp"
    output_Dir = "output/1812/27-1 (4).jpg"
    mask_Dir = "test/test_mask/theme_mask.bmp"
    checkpoint_Dir = "logs/log1812/model.ckpt-82500"

    # get mask image
    image_size = 224

    test_img = tf.read_file(test_Dir)
    l_channel, ab_channel = get_lab_channel(test_img, image_size, "jpg")

    theme_img = tf.read_file(theme_Dir, "bmp")
    theme = get_theme_channels(theme_img, "bmp")

    mask_img = tf.read_file(mask_Dir)
    mask = get_theme_mask(mask_img, "bmp")

    ab_channel = (ab_channel + 128) / 255

    theme_input = tf.concat([theme, mask], 3)

    ab_out = model.built_network(ab_channel, theme_input)

    # load ckpt file, load the model
    logs_dir = 'F:/Deep Learning/Code/colorization/logs/log1228'
    saver = tf.train.Saver()

    sess = tf.Session()
    print('载入检查点...')

    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt.model_checkpoint_path = checkpoint_Dir
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    l_channel = tf.cast(l_channel, tf.float64)
    ab_out = tf.cast(ab_out, tf.float64)

    l, ab = sess.run([l_channel, ab_out])
    l = l[0]
    ab = ab[0]
    l = l * 100
    ab = ab * 255 - 128
    img_out = np.concatenate([l, ab], 2)
    img_out = color.lab2rgb(img_out)
    plt.imsave(output_Dir, img_out)


run_training()
#test_one_image()
#test_theme_image()
# test_batch_image()
