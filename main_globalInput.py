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

#global theme input, use 3 resnet, 5 color theme
def run_training():
    train_dir = "G:\\Database\\ColoredData\\new_colorimage1"
    theme_index_dir = "G:\\Database\\ColoredData\\ColorMap5_image4"
    image_index_dir = "G:\\Database\\ColoredData\\new_colorimage4"
    theme_dir = "G:\\Database\\ColoredData\\colorImages4_5theme"
    #theme_mask_dir = "G:\\Database\\ColorThemeMask5"

    logs_dir = "F:\\Project_Yang\\Code\\mainProject\\logs\\log_global\\image loss 0.2"
    result_dir = "results/global/"

    # 获取输入
    image_list = input_data.get_themeInput_list(train_dir, theme_dir, theme_index_dir, image_index_dir)
    train_batch, theme_batch, theme_index_batch, theme_mask_batch, image_index_batch = input_data.get_themeObj_batch(image_list, BATCH_SIZE, CAPACITY)

    #rgb_to_lab
    train_batch = tf.cast(train_batch, tf.float64)
    image_index_batch = tf.cast(image_index_batch, tf.float64)
    theme_batch = tf.cast(theme_batch, tf.float64)
    theme_index_batch = tf.cast(theme_index_batch, tf.float64)

    train_lab_batch = tf.cast(input_data.rgb_to_lab(train_batch), tf.float32)
    theme_lab_batch = tf.cast(input_data.rgb_to_lab(theme_batch), tf.float32)
    index_lab_batch = tf.cast(input_data.rgb_to_lab(image_index_batch), tf.float32)
    themeIndex_lab_batch = tf.cast(input_data.rgb_to_lab(theme_index_batch), tf.float32)

    #do + - * / before normalization

    #normalization
    image_l_batch = tf.reshape(train_lab_batch[:, :, :, 0] / 100, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_ab_batch = (train_lab_batch[:, :, :, 1:] + 128) / 255
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128) / 255
    theme_l_batch = (theme_lab_batch[:, :, :, 0:1] + 128) / 255
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255
    themeIndex_ab_batch = (themeIndex_lab_batch[:, :, :, 1:] + 128) / 255

    #input batches
    theme_input = tf.concat([theme_ab_batch, theme_mask_batch], 3)

    #concat image_ab and sparse_ab as input
    out_ab_batch = model.new_built_network(image_ab_batch, theme_input)

    image_l_batch = tf.cast(image_l_batch, tf.float64)
    theme_lab_batch = tf.cast(theme_lab_batch, tf.float64)

    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss, index_loss, color_loss, image_loss = model.whole_loss(out_ab_batch, index_ab_batch, themeIndex_ab_batch, image_ab_batch)
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

            _, tra_loss, ind_loss, col_loss, img_loss = sess.run([train_op, train_loss, index_loss, color_loss, image_loss])
            tra_rmse, tra_psnr = sess.run([train_rmse, train_psnr])

            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,  loss: %g,  index_loss: %g,  color_loss: %g, image_loss: %g,  rmse: %g,  psnr: %g" % (step, tra_loss, ind_loss, col_loss, img_loss, tra_rmse, tra_psnr))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 2000 == 0:
                l, ab, ab_index, ab_out, theme_lab, colored = sess.run(
                    [image_l_batch, image_ab_batch, index_ab_batch, out_ab_batch, theme_lab_batch, themeIndex_ab_batch])
                l = l[0] * 100
                ab = ab[0] * 255 - 128
                ab_index = ab_index[0] * 255 - 128
                ab_out = ab_out[0] * 255 - 128
                colored = colored[0] * 255 - 128

                img_in = np.concatenate([l, ab], 2)
                img_in = color.lab2rgb(img_in)
                img_out = np.concatenate([l, ab_out], 2)
                img_out = color.lab2rgb(img_out)
                img_index = np.concatenate([l, ab_index], 2)
                img_index = color.lab2rgb(img_index)
                img_colored = np.concatenate([l, colored], 2)
                img_colored = color.lab2rgb(img_colored)
                theme = color.lab2rgb(theme_lab[0])

                plt.subplot(5, 4, 1), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(5, 4, 2), plt.imshow(ab[:, :, 0], 'gray')
                plt.subplot(5, 4, 3), plt.imshow(ab[:, :, 1], 'gray')
                plt.subplot(5, 4, 4), plt.imshow(img_in)

                plt.subplot(5, 4, 5), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(5, 4, 6), plt.imshow(ab_out[:, :, 0], 'gray')
                plt.subplot(5, 4, 7), plt.imshow(ab_out[:, :, 1], 'gray')
                plt.subplot(5, 4, 8), plt.imshow(img_out)

                plt.subplot(5, 4, 9), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(5, 4, 10), plt.imshow(ab_index[:, :, 0], 'gray')
                plt.subplot(5, 4, 11), plt.imshow(ab_index[:, :, 1], 'gray')
                plt.subplot(5, 4, 12), plt.imshow(img_index)

                plt.subplot(5, 4, 13), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(5, 4, 14), plt.imshow(colored[:, :, 0], 'gray')
                plt.subplot(5, 4, 15), plt.imshow(colored[:, :, 1], 'gray')
                plt.subplot(5, 4, 16), plt.imshow(img_colored)

                plt.subplot(5, 4, 17), plt.imshow(theme)
                plt.savefig(result_dir + str(step) + "_image.png")
                plt.show()


                plt.figure(figsize=(8,8))
                axes1 = plt.subplot(231)
                axes1.scatter(ab[:, :, 0], ab[:, :, 1],alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('input images')

                axes2 = plt.subplot(232)
                axes2.scatter(ab_out[:, :, 0], ab_out[:, :, 1],alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('output images')

                axes3 = plt.subplot(233)
                axes3.scatter(ab_index[:, :, 0], ab_index[:, :, 1], alpha=0.5, edgecolor='white', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('index images')

                axes4 = plt.subplot(234)
                axes4.scatter(colored[:, :, 0], colored[:, :, 1], alpha=0.5, edgecolor="white", s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                plt.title('colored images')

                axes5 = plt.subplot(235)
                part1 = axes5.scatter(ab[:, :, 0], ab[:, :, 1], alpha=0.5, edgecolor='white', label='image_in', s=8)
                part2 = axes5.scatter(ab_index[:, :, 0], ab_index[:, :, 1], alpha=0.5, edgecolor='white', label='image_index', c='g', s=8)
                part3 = axes5.scatter(colored[:, :, 0], colored[:, :, 1], alpha=0.5,edgecolor='white', label='image_out', c = 'y', s=8)
                part4 = axes5.scatter(ab_out[:, :, 0], ab_out[:, :, 1], alpha=0.5, edgecolor='white', label='image_out',c='r', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                axes4.legend((part1, part2, part3, part4), ('input', 'index', 'colored', 'output'))
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
    height = image_size[0]
    width = image_size[1]
    l_channel = tf.image.resize_images(l_channel, [height, width])
    l_channel = tf.cast(l_channel, tf.float64) / 255.0
    l_channel = input_data.rgb_to_lab(l_channel)
    l_channel = tf.cast(l_channel, tf.float32)
    ab_channel = l_channel[:, :, 1:]
    #ab_channel = (l_channel[:, :, 1:] + 128) / 255.0
    l_channel = l_channel[:, :, 0] / 100.0
    l_channel = tf.reshape(l_channel, [height, width, 1])
    ab_channel = tf.reshape(ab_channel, [1, height, width, 2])
    l_channel = tf.reshape(l_channel, [1, height, width, 1])
    #lab_channel = tf.reshape(lab_channel, [1, height, width, 3])
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
    theme_img = tf.cast(theme_img, tf.float64) / 255.0
    theme_img = input_data.rgb_to_lab(theme_img)
    theme_img = tf.cast(theme_img, tf.float32)
    theme_ab = (theme_img[:, :, 1:] + 128) / 255
    theme_ab = tf.reshape(theme_ab, [1, 1, 5, 2])
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



def test_theme_image():
    # 15,18,20,21,27
    test_Dir = "test/test_images2/20.jpg"
    theme_Dir = "test/test_images2/theme_566.bmp"
    output_Dir = "output/global/image_loss2/20-566.jpg"
    mask_Dir = "test/test_mask/theme_mask.bmp"
    checkpoint_Dir = "logs/log_global/image_loss2/model.ckpt-60000"

    sess = tf.Session()

    test_img = tf.read_file(test_Dir)
    image_size = get_imageSize(test_img, sess, "jpg")
    l_channel, ab_channel = get_lab_channel(test_img, image_size, "jpg")

    theme_img = tf.read_file(theme_Dir, "bmp")
    theme = get_theme_channels(theme_img, "bmp")
    #theme = tf.zeros([1, 1, 5, 2], dtype=tf.float32)

    mask_img = tf.read_file(mask_Dir)
    mask = get_theme_mask(mask_img, "bmp")

    ab_channel = (ab_channel + 128) / 255

    theme_input = tf.concat([theme, mask], 3)

    ab_out = model.new_built_network(ab_channel, theme_input)

    # load ckpt file, load the model
    logs_dir = 'F:/Deep Learning/Code/colorization/logs/log1228'
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


#run_training()
#test_one_image()
test_theme_image()
# test_batch_image()
