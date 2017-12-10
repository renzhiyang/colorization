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

BATCH_SIZE = 5
CAPACITY = 1000     # 队列容量
MAX_STEP = 150000

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 指定GPU


def run_training1():
    train_dir = "F:\\Project_Yang\\Database\\database_new\\training_image"
    #sparse_dir = "F:\\Project_Yang\\Database\\database_new\\sparse_image"
    index_dir = "F:\\Project_Yang\\Database\\database_new\\index_image"
    mask_dir = "F:\\Project_Yang\\Database\\database_new\\mask_image"


    #train_dir = "F:\\Project_Yang\\Database\\database_test\\train_images"
    #sparse_dir = "F:\\Project_Yang\\Database\\database_test\\sparse_images"
    #index_dir = "F:\\Project_Yang\\Database\\database_test\\index_images"
    #mask_dir = "F:\\Project_Yang\\Database\\database_test\\mask_images"
    logs_dir = "F:\\Project_Yang\\Code\\mainProject\\logs\\log1210"

    # 获取输入
    image_list = input_data.get_image_list2(train_dir, mask_dir, index_dir)
    l_batch, ab_batch, lab_batch, index_ab_batch, mask_batch_2channels = input_data.get_batch2(image_list, BATCH_SIZE, CAPACITY)

    sparse_ab_batch = ab_batch * mask_batch_2channels
    replace_image = (ab_batch - sparse_ab_batch) + (index_ab_batch * mask_batch_2channels)
    mask_batch = mask_batch_2channels[:, :, :, 0]
    mask_batch = tf.reshape(mask_batch, [BATCH_SIZE, 224, 224, 1])

    out_ab_batch = model.built_network(replace_image, mask_batch)
    sess = tf.Session()

    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_loss = model.whole_loss(out_ab_batch, index_ab_batch, mask_batch_2channels, sparse_ab_batch)
    train_rmse, train_psnr = model.get_PSNR(out_ab_batch, index_ab_batch)
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

            if step % 10000 == 0:
                l, ab, ab_index, ab_out, ab_replace = sess.run([l_batch, ab_batch, index_ab_batch, out_ab_batch, replace_image])
                #replace_ab = sess.run(replace_image)
                l = l[0]
                ab = ab[0]
                ab_index = ab_index[0]
                ab_out = ab_out[0]
                replace_ab = replace_ab[0]

                print([l[:, :, 0].min(), l[:, :, 0].max()])
                print([ab_out[:, :, 0].min(), ab_out[:, :, 0].max()])
                print([ab_out[:, :, 1].min(), ab_out[:, :, 1].max()])

                l = l * 100
                ab = ab * 255 - 128
                ab_out = ab_out * 255 - 128
                ab_index = ab_index * 255 -128
                replace_ab = replace_ab * 255 -128
                img_in = np.concatenate([l, ab], 2)
                img_in = color.lab2rgb(img_in)
                img_out = np.concatenate([l, ab_out], 2)
                img_out = color.lab2rgb(img_out)
                img_index = np.concatenate([l, ab_index], 2)
                img_index = color.lab2rgb(img_index)
                replace_in = np.concatenate([l, replace_ab], 2)
                replace_in = color.lab2rgb(replace_in)


                #print([l[:, :, 0].min(), l[:, :, 0].max()])
                #print([ab_out[:, :, 0].min(), ab_out[:, :, 0].max()])
                #print([ab_out[:, :, 1].min(), ab_out[:, :, 1].max()])
                #print()
                plt.subplot(4, 4, 1), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 2), plt.imshow(ab[:, :, 0], 'gray')
                plt.subplot(4, 4, 3), plt.imshow(ab[:, :, 1], 'gray')
                plt.subplot(4, 4, 4), plt.imshow(img_in)

                plt.subplot(4, 4, 5), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 6), plt.imshow(replace_ab[:, :, 0], 'gray')
                plt.subplot(4, 4, 7), plt.imshow(replace_ab[:, :, 1], 'gray')
                plt.subplot(4, 4, 8), plt.imshow(replace_in)

                plt.subplot(4, 4, 9), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 10), plt.imshow(ab_out[:, :, 0], 'gray')
                plt.subplot(4, 4, 11), plt.imshow(ab_out[:, :, 1], 'gray')
                plt.subplot(4, 4, 12), plt.imshow(img_out)

                plt.subplot(4, 4, 13), plt.imshow(l[:, :, 0], 'gray')
                plt.subplot(4, 4, 14), plt.imshow(ab_index[:, :, 0], 'gray')
                plt.subplot(4, 4, 15), plt.imshow(ab_index[:, :, 1], 'gray')
                plt.subplot(4, 4, 16), plt.imshow(img_index)
                plt.show()


    except tf.errors.OutOfRangeError:
        print("Done.")
    finally:
        coord.request_stop()

    # 等待线程结束
    coord.join(threads=threads)
    sess.close()


def get_lab_channel(image, image_size, type):
    if type == "jpg":
        l_channel = tf.image.decode_jpeg(image, channels = 3)
    if type == "bmp":
        l_channel = tf.image.decode_bmp(image, channels = 3)

    l_channel = tf.image.resize_images(l_channel, [image_size, image_size])
    l_channel = tf.cast(l_channel, tf.float64) / 255.0
    l_channel = input_data.rgb_to_lab(l_channel)
    l_channel = tf.cast(l_channel, tf.float32)
    ab_channel = (l_channel[:, :, 1:] + 128) / 255.0
    l_channel = l_channel[:, :, 0] / 100.0
    l_channel = tf.reshape(l_channel, [image_size, image_size, 1])
    ab_channel = tf.reshape(ab_channel, [1, image_size, image_size, 2])
    l_channel = tf.reshape(l_channel, [1, image_size, image_size, 1])
    #lab_channel = tf.reshape(lab_channel, [1, image_size, image_size, 3])
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
    test_Dir = "test_images/test (2).bmp"
    sparse_Dir = "test_sparses/blue.bmp"
    output_Dir = "output1210/2-blue.jpg"
    mask_Dir = "test_mask/blue.bmp"
    checkpoint_Dir = "log_1208/model.ckpt-149999"

    #get mask image
    image_size = 224
    get_mask(sparse_Dir, mask_Dir, image_size)

    test_img = tf.read_file(test_Dir)
    l_channel, ab_channel = get_lab_channel(test_img, image_size, "jpg")

    sparse_img = tf.read_file(sparse_Dir)
    l_sparse, ab_sparse = get_lab_channel(sparse_img, image_size, "bmp")

    mask_img = tf.read_file(mask_Dir)
    mask_one_channel, mask_two_channels = get_mask_channels(mask_img, image_size)

    replace_ab_image = ab_channel - ( ab_channel * mask_two_channels) + ab_sparse


    ab_out = model.built_network(replace_ab_image, mask_one_channel)

    #load ckpt file, load the model
    logs_dir = 'F:/Project_Yang/Code/mainProject/log1208'
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

    l_inputImage, ab_inputImage,  ab_outImage = sess.run([l_channel, ab_channel, ab_out])
    l_inputImage = l_inputImage[0]
    ab_inputImage = ab_inputImage[0]
    ab_outImage = ab_outImage[0]

    l_inputImage = l_inputImage * 100
    ab_inputImage = ab_inputImage * 255 - 128
    ab_outImage = ab_outImage * 255 - 128

    image_in = np.concatenate([l_inputImage, ab_inputImage], 2)
    image_out = np.concatenate([l_inputImage, ab_outImage], 2)
    image_in = color.lab2rgb(image_in)
    image_out = color.lab2rgb(image_out)

    print(ab_inputImage[:, :, 0].min(), ab_inputImage[:, :, 0].max())
    print(l_inputImage[:, :, 0].min(), l_inputImage[:, :, 0].max())

    plt.subplot(2, 4, 1), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(2, 4, 2), plt.imshow(ab_inputImage[:, :, 0], 'gray')
    plt.subplot(2, 4, 3), plt.imshow(ab_inputImage[:, :, 1], 'gray')
    plt.subplot(2, 4, 4), plt.imshow(image_in, 'gray')

    plt.subplot(2, 4, 5), plt.imshow(l_inputImage[:, :, 0], 'gray')
    plt.subplot(2, 4, 6), plt.imshow(ab_outImage[:, :, 0], 'gray')
    plt.subplot(2, 4, 7), plt.imshow(ab_outImage[:, :, 1], 'gray')
    plt.subplot(2, 4, 8), plt.imshow(image_out, 'gray')
    plt.show()


    plt.imsave(output_Dir, image_out)



def eval_one_image1():
    # sparse_name: blueLine, none, red&blue, red&blue2, redLine
    test_Dir = "test_images/5.bmp"
    sparse_Dir = "test_sparses/redLine.jpg"
    output_Dir = "out_images1207/5-redLine-142500.jpg"
    checkpoint_Dir = "F:/Project_Yang/Code/mainProject/log_1208/model.ckpt-149999"

    image_size = 224
    test_image1 = tf.read_file('test_images2/1_2.jpg')
    l_channel, ab_channel = get_lab_channel(test_image1, image_size)
    test_input = tf.concat([l_channel, ab_channel], 3)

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

    print([l[0, :, :, 0].min(), l[0, :, :, 0].max()])
    print([ab[0, :, :, 0].min(), ab[0, :, :, 0].max()])
    print([ab[0, :, :, 1].min(), ab[0, :, :, 1].max()])
    print()

    plt.imsave('out_images/1_2_testout_' + fix + '.bmp', img_out1)


#run_training1()
test_one_image()
# test_batch_image()
