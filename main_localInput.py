# training.py  训练、测试
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import input_data
import model_localInput as model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 10
CAPACITY = 1000     # 队列容量
MAX_STEP = 150000

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU

#local input
def run_training():
    train_dir = "F:\\Database\\ColoredData\\new_colorimage1"
    index_dir = "F:\\Database\\ColoredData\\new_colorimage4"
    mask_dir = "F:\\Database\\ColoredData\\newSparseMask"
    sparse_dir = "F:\\Database\\ColoredData\\newSparse"
    logs_dir = "E:\\Project_Yang\\Code\\logs\\local\\local3"
    result_dir = "results/local/local3/"

    # 获取输入
    image_list = input_data.get_local_list(train_dir, sparse_dir, mask_dir, index_dir)
    train_rgb_batch, sparse_rgb_batch, mask_2channels_batch, index_rgb_batch\
        = input_data.get_local_batch(image_list, BATCH_SIZE, CAPACITY)

    train_lab_batch = tf.cast(input_data.rgb_to_lab(train_rgb_batch), tf.float32)
    sparse_lab_batch = tf.cast(input_data.rgb_to_lab(sparse_rgb_batch), tf.float32)
    index_lab_batch = tf.cast(input_data.rgb_to_lab(index_rgb_batch), tf.float32)
    mask_2channels_batch = tf.cast(mask_2channels_batch, tf.float32)

    #do '+ - * /' before normalization
    train_l_batch = train_lab_batch[:, :, :, 0:1] / 100
    train_ab_batch = (train_lab_batch[:, :, :, 1:] + 128) / 255
    sparse_l_batch = sparse_lab_batch[:, :, :, 0:1] / 100
    sparse_ab_batch = (sparse_lab_batch[:, :, :, 1:] + 128) / 255
    index_l_batch = index_lab_batch[:, :, :, 0:1] / 100
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255
    sparse_input = tf.concat([sparse_ab_batch, mask_2channels_batch[:, :, :, 0:1]], 3)

    #concat image_ab and sparse_ab as input
    out_ab_batch = model.built_network(train_ab_batch, sparse_input)
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss, separate_loss = model.whole_loss(out_ab_batch, index_ab_batch, train_ab_batch, mask_2channels_batch)
    train_rmse, train_psnr = model.get_PSNR(out_ab_batch, index_ab_batch)
    train_op = model.training(train_loss, global_step)

    train_l_batch = tf.cast(train_l_batch, tf.float64)
    index_l_batch = tf.cast(index_l_batch, tf.float64)
    #lab_batch = tf.cast(lab_batch, tf.float64)

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

            _, tra_loss, sep_loss = sess.run([train_op, train_loss, separate_loss])
            tra_rmse, tra_psnr = sess.run([train_rmse, train_psnr])

            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,    loss: %g,  index_loss: %g,   RMSE: %g,   PSNR: %g" % (step, tra_loss, sep_loss[0], tra_rmse, tra_psnr))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0:
                l, ab, ab_index, ab_out = sess.run(
                    [train_l_batch, train_ab_batch, index_ab_batch, out_ab_batch])
                l = l[0]
                ab = ab[0]
                ab_index = ab_index[0]
                ab_out = ab_out[0]

                l = l * 100
                ab = ab * 255 - 128
                ab_out = ab_out * 255 - 128
                ab_index = ab_index * 255 -128


                img_in = np.concatenate([l, ab], 2)
                img_in = color.lab2rgb(img_in)
                img_out = np.concatenate([l, ab_out], 2)
                img_out = color.lab2rgb(img_out)
                img_index = np.concatenate([l, ab_index], 2)
                img_index = color.lab2rgb(img_index)

                plt.subplot(3, 4, 1), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(3, 4, 2), plt.imshow(ab[:, :, 0], 'gray')
                plt.subplot(3, 4, 3), plt.imshow(ab[:, :, 1], 'gray')
                plt.subplot(3, 4, 4), plt.imshow(img_in)

                plt.subplot(3, 4, 5), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(3, 4, 6), plt.imshow(ab_out[:, :, 0], 'gray')
                plt.subplot(3, 4, 7), plt.imshow(ab_out[:, :, 1], 'gray')
                plt.subplot(3, 4, 8), plt.imshow(img_out)

                plt.subplot(3, 4, 9), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(3, 4, 10), plt.imshow(ab_index[:, :, 0], 'gray')
                plt.subplot(3, 4, 11), plt.imshow(ab_index[:, :, 1], 'gray')
                plt.subplot(3, 4, 12), plt.imshow(img_index)
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
                part2 = axes4.scatter(ab_index[:, :, 0], ab_index[:, :, 1], alpha=0.5, edgecolor='white', label='image_index', c='g', s=8)
                part3 = axes4.scatter(ab_out[:, :, 0], ab_out[:, :, 1], alpha=0.5, edgecolor='white', label='image_out', c='r', s=8)
                plt.xlabel('a')
                plt.ylabel('b')
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                axes4.legend((part1, part2, part3), ('input', 'index', 'output'))
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

def get_sparse_mask(sparse_mask, image_size, type):
    if type == "jpg":
        sparse_mask = tf.image.decode_jpeg(sparse_mask, channels = 3)
    if type == "bmp":
        sparse_mask = tf.image.decode_bmp(sparse_mask, channels = 3)
    sparse_mask = tf.image.resize_images(sparse_mask, [image_size[0], image_size[1]])
    sparse_mask = tf.cast(sparse_mask, tf.float32) / 255
    sparse_mask = tf.reshape(sparse_mask, [1, image_size[0], image_size[1], 3])
    sparse_mask = sparse_mask[:, :, :, 0:1]
    return sparse_mask


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
    # sparse_name: blueLine, none, red&blue, red&blue2, redLine
    test_Dir = "test/test_images/20.jpg"
    sparse_Dir = "test/test_images2/local_20_6.bmp"
    sparse_mask_Dir = "test/test_images2/mask_20_6.bmp"
    output_Dir = "output/local/local2/20-6.jpg"
    checkpoint_Dir = "logs/log_local/local2/model.ckpt-75000"

    sess = tf.Session()

    test_img = tf.read_file(test_Dir)
    image_size = get_imageSize(test_img, sess, "jpg")
    l_channel, ab_channel = get_lab_channel(test_img, image_size, "jpg")

    sparse_img = tf.read_file(sparse_Dir)
    sparse_l, sparse_ab = get_lab_channel(sparse_img, image_size, "bmp")

    sparse_mask = tf.read_file(sparse_mask_Dir)
    sparse_mask = get_sparse_mask(sparse_mask, image_size, "bmp")

    ab_channel = (ab_channel + 128) / 255
    sparse_ab = (sparse_ab + 128) / 255

    sparse_input = tf.concat([sparse_ab, sparse_mask[:, :, :, 0:1]], 3)

    ab_out = model.built_network(ab_channel, sparse_input)

    # load ckpt file, load the model
    logs_dir = 'F:/Deep Learning/Code/colorization/logs/log_local/local2/'
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

    sparse_l = tf.cast(sparse_l, tf.float64)
    sparse_ab = tf.cast(sparse_ab, tf.float64)
    sparse_l, sparse_ab, out_l, out_ab = sess.run([sparse_l, sparse_ab, l_channel, ab_out])
    sparse_l = sparse_l[0]
    sparse_ab = sparse_ab[0]
    out_ab = out_ab[0]
    out_l = out_l[0]

    sparse_l = sparse_l * 100
    out_l = out_l * 100
    sparse_ab = sparse_ab * 255 - 128
    out_ab = out_ab * 255 - 128
    sparse = np.concatenate([sparse_l, sparse_ab], 2)
    out = np.concatenate([out_l, out_ab], 2)
    out = color.lab2rgb(out)
    sparse = color.lab2rgb(sparse)
    plt.subplot(1,3,1), plt.imshow(out_ab[:, :, 0], 'gray')
    plt.subplot(1,3,2), plt.imshow(out_ab[:, :, 1], 'gray')
    plt.subplot(1,3,3), plt.imshow(out, 'gray')
    plt.show()



    l, ab = sess.run([l_channel, ab_out])
    l = l[0]
    ab = ab[0]
    l = l * 100
    ab = ab * 255 - 128
    img_out = np.concatenate([l, ab], 2)
    img_out = color.lab2rgb(img_out)
    #if not os.path.exists(output_Dir):
    #    os.makedirs(output_Dir)
    plt.imsave(output_Dir, img_out)


run_training()
#test_theme_image()
#test_one_image()
# test_batch_image()
