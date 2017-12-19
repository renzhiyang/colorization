# training.py  训练、测试
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import input_data
import model1207 as model
from math import isnan
from matplotlib import pyplot as plt
import skimage.color as color
## import cv2

BATCH_SIZE = 10
CAPACITY = 1000     # 队列容量
MAX_STEP = 150000

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU


def run_training():
    train_dir = "F:\\Project_Yang\\Database\\database_new\\ColorMap_all"
    #sparse_dir = "F:\\Project_Yang\\Database\\database_new\\sparse_image"
    #index_dir = "F:\\Project_Yang\\Database\\database_new\\ColorMap_all"
    mask_dir = "F:\\Project_Yang\\Database\\database_new\\mask_image"
    logs_dir = "F:\\Project_Yang\\Code\\mainProject\\logs\\log1218"
    result_dir = "results/1218/"

    # 获取输入
    image_list = input_data.get_image_list1219(train_dir, mask_dir)
    l_batch, ab_batch, mask_batch, mask_batch_2channels = input_data.get_batch_1219(image_list, BATCH_SIZE, CAPACITY)

    sparse_ab_batch = ab_batch * mask_batch_2channels

    #do '+ - * /' before normalization
    l_batch = l_batch / 100
    ab_batch = (ab_batch + 128) / 255
    index_ab_batch = (index_ab_batch + 128) / 255
    sparse_ab_batch = (sparse_ab_batch + 128) / 255
    replace_image = (replace_image + 128) / 255

    #concat images
    #input_batch = tf.concat([ab_batch, sparse_ab_batch], 3)
    #mask_batch = mask_batch_2channels[:, :, :, 0]
    #mask_batch = tf.reshape(mask_batch, [BATCH_SIZE, 224, 224, 1])

    #gray image input
    gray_input = tf.concat([l_batch, sparse_ab_batch], 3)

    #concat image_ab and sparse_ab as input
    out_ab_batch = model.built_network1212(gray_input, mask_batch)
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss = model.gray_colorization_loss(out_ab_batch, ab_batch)
    train_rmse, train_psnr = model.get_PSNR(out_ab_batch, index_ab_batch)
    train_op = model.training(train_loss, global_step)

    l_batch = tf.cast(l_batch, tf.float64)
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

            _, tra_loss = sess.run([train_op, train_loss])
            tra_rmse, tra_psnr = sess.run([train_rmse, train_psnr])

            if isnan(tra_loss):
                print('Loss is NaN.')
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)
                exit(-1)
            if step % 100 == 0:     # 及时记录MSE的变化
                merged = sess.run(summary_op)
                train_writer.add_summary(merged, step)
                print("Step: %d,    loss: %g,     RMSE: %g,     PSNR: %g" % (step, tra_loss, tra_rmse, tra_psnr))
            if step % (MAX_STEP/20) == 0 or step == MAX_STEP-1:     # 保存20个检查点
                checkpoint_path = os.path.join(logs_dir, "model.ckpt")
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 2000 == 0:
                l, ab, ab_index, ab_out = sess.run(
                    [l_batch, ab_batch, index_ab_batch, out_ab_batch])
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


def test_one_image():
    # sparse_name: blueLine, none, red&blue, red&blue2, redLine
    test_Dir = "test/test_images/iizuka.jpg"
    sparse_Dir = "test/test_sparses/iizuka.bmp"
    output_Dir = "output/output1215/iizuka2.jpg"
    mask_Dir = "test/test_mask/iizuka.bmp"
    checkpoint_Dir = "logs/log1215/model.ckpt-149999"

    #get mask image
    image_size = 224
    get_mask(sparse_Dir, mask_Dir, image_size)

    test_img = tf.read_file(test_Dir)
    l_channel, ab_channel = get_lab_channel(test_img, image_size, "jpg")

    sparse_img = tf.read_file(sparse_Dir)
    l_sparse, ab_sparse = get_lab_channel(sparse_img, image_size, "bmp")

    mask_img = tf.read_file(mask_Dir)
    mask_one_channel, mask_two_channels = get_mask_channels(mask_img, image_size)

    replace_ab_image = (ab_channel - ab_channel * mask_two_channels) + ab_sparse
    #replace_ab_image = ab_channel - (ab_channel * mask_two_channels)
    middle_ab_image = ab_channel - (ab_channel * mask_two_channels)

    replace_ab_image = (replace_ab_image + 128) / 255
    ab_sparse = (ab_sparse + 128) / 255
    middle_ab_image = (middle_ab_image + 128) / 255
    ab_channel = (ab_channel + 128) / 255
    input_batch = tf.concat([ab_channel, ab_sparse], 3)

    ab_out = model.built_network1212(input_batch, mask_one_channel)

    #load ckpt file, load the model
    logs_dir = 'F:/Project_Yang/Code/mainProject/logs/log1208'
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
    ab_channel = tf.cast(ab_channel, tf.float64)
    ab_out = tf.cast(ab_out, tf.float64)

    l_inputImage, ab_inputImage,  ab_outImage, ab_replace, ab_spar, ab_middle = sess.run([l_channel, ab_channel, ab_out, replace_ab_image, ab_sparse, middle_ab_image])
    l_inputImage = l_inputImage[0]
    ab_inputImage = ab_inputImage[0]
    ab_outImage = ab_outImage[0]
    ab_replace = ab_replace[0]
    ab_spar = ab_spar[0]
    ab_middle = ab_middle[0]

    print(ab_spar[:, :, 0].min(), ab_spar[:, :, 0].max())
    print(ab_outImage[:, :, 0].min(), ab_outImage[:, :, 0].max())
    print(ab_middle[:, :, 0].min(), ab_middle[:, :, 0].max())

    l_inputImage = l_inputImage * 100
    ab_inputImage = ab_inputImage * 255 - 128
    ab_outImage = ab_outImage * 255 - 128
    ab_replace = ab_replace * 255 - 128
    ab_spar = ab_spar * 255 - 128
    ab_middle = ab_middle * 255 - 128

    image_in = np.concatenate([l_inputImage, ab_inputImage], 2)
    image_out = np.concatenate([l_inputImage, ab_outImage], 2)
    image_replace = np.concatenate([l_inputImage, ab_replace], 2)
    image_sparse = np.concatenate([l_inputImage, ab_spar], 2)
    image_in = color.lab2rgb(image_in)
    image_out = color.lab2rgb(image_out)
    image_replace = color.lab2rgb(image_replace)
    image_sparse = color.lab2rgb(image_sparse)

    plt.subplot(4, 4, 1), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 2), plt.imshow(ab_inputImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 3), plt.imshow(ab_inputImage[:, :, 1], 'gray')
    plt.subplot(4, 4, 4), plt.imshow(image_in, 'gray')

    plt.subplot(4, 4, 5), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 6), plt.imshow(ab_outImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 7), plt.imshow(ab_outImage[:, :, 1], 'gray')
    plt.subplot(4, 4, 8), plt.imshow(image_out, 'gray')

    plt.subplot(4, 4, 9), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 10), plt.imshow(ab_replace[:, :, 0], 'gray')
    plt.subplot(4, 4, 11), plt.imshow(ab_replace[:, :, 1], 'gray')
    plt.subplot(4, 4, 12), plt.imshow(image_replace, 'gray')

    plt.subplot(4, 4, 13), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(4, 4, 14), plt.imshow(ab_spar[:, :, 0], 'gray')
    plt.subplot(4, 4, 15), plt.imshow(ab_spar[:, :, 1], 'gray')
    plt.subplot(4, 4, 16), plt.imshow(image_sparse, 'gray')
    plt.show()


    plt.imsave(output_Dir, image_out)


run_training()
#test_one_image()
# test_batch_image()
