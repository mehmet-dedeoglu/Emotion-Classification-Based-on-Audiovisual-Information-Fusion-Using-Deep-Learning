from __future__ import print_function
import pylab
import imageio
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from MultimediaCNN import AudioClassifier, FrameClassifier, OutputClassifier, FrameClassifier64


# The class definition for inputs; defines the values related to inputs.
class InputClass(object):
    pass


# Decode video frames from video paths.
def decode_frames(_filepathv):
    video = imageio.get_reader(_filepathv,  'ffmpeg')
    image0 = video.get_data(0)

    shape = image0.shape
    #frame_slices = np.zeros([shape[0], shape[1], shape[2], 90], dtype=np.uint8)
    #frame_slices[:, :, :, 0] = image0
    #for num in np.arange(1, 90):
    #    frame_slices[:, :, :, num] = video.get_data(num)

    frame_slices = np.zeros([shape[0], shape[1], 16, 1], dtype=np.uint8)
    frame_slices[:, :, 0, 0] = image0[:, :, 0]
    for num in np.arange(1, 16):
        frame_slices[:, :, num, 0] = video.get_data(4*num)[:, :, 0]

    return frame_slices

def decode_video_frames(filepathv):
    image_frame_decoder = lambda fp: decode_frames(fp)

    image_frames = tf.py_func(image_frame_decoder,
                              [filepathv], tf.uint8, stateful=False)
    image_frames.set_shape([720, 1280, 16, 1])
    im_final = tf.reshape(tf.image.resize_images
                          (image_frames[:, :, 0, :], [64, 64]),
                          [64, 64, 1, 1])
    for im in np.arange(1, 16):
        im_final = tf.concat([im_final,
                              tf.reshape(tf.image.resize_images
                                         (image_frames[:, :, im, :], [64, 64]),
                                         [64, 64, 1, 1])], axis=2)
    im_final = im_final / 255.0
    print(im_final.shape)
    return im_final


def decode_labels(_filepathl):
    folder = os.path.basename(os.path.dirname(_filepathl))
    label = 0
    if folder == 'Happy':
        label = 1
    elif folder == 'Angry':
        label = 2
    elif folder == 'Calm':
        label = 3
    elif folder == 'Disgust':
        label = 4
    elif folder == 'Fearful':
        label = 5
    elif folder == 'Neutral':
        label = 6
    elif folder == 'Surprised':
        label = 7
    return label


def decode_video_labels(filepathl):
    image_label_decoder = lambda filepathl: decode_labels(filepathl)
    image_labels = tf.py_func(image_label_decoder,
                              [filepathl], tf.int64, stateful=False)
    #image_labels.set_shape([None])
    return image_labels


def decode_audio(fpsa, first_slice=True, _slice=131072):
    audio_binary = tf.read_file(fpsa)
    waveform = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format='mp4',
                                              samples_per_second=48000,
                                              channel_count=1)
    waveform = waveform / tf.reduce_max(tf.abs(waveform))
    if first_slice:
        waveform = waveform[:_slice, :]
    waveform.set_shape([_slice, 1])
    #wave = tf.data.Dataset.from_tensor_slices(waveform)
    return waveform


# Create a Tensorflow training iterator. First, create a dataset of filepaths.
# Training session calls batches according to filepaths. At each call, batch is
# loaded, split into audio and image frames.
def load_batch(fps, batch_size):
    # Create a dataset of filepaths.
    dataset = tf.data.Dataset.from_tensor_slices(fps)

    # Preprocess audio and video frames, and create a dataset of labels.
    audio_dataset = dataset.map(decode_audio)
    label_dataset = dataset.map(decode_video_labels)
    frame_dataset = dataset.map(decode_video_frames)

    # Connect all three datasets together for batching datasets in the same order.
    dataset_zipped = tf.data.Dataset.zip((frame_dataset, label_dataset,
                                          dataset, audio_dataset))

    # Shuffle dataset for better training.
    # Shuffling takes too long. So, for testing, use the second line that doesn't
    # shuffle the datasets. For real training, use the first line.
    dataset_zipped = dataset_zipped.shuffle(buffer_size=len(fps)).repeat().shuffle(buffer_size=len(fps))
    #dataset_zipped = dataset_zipped.repeat()

    # Create an iterator for pipelining the dataset into the
    iterz = dataset_zipped.batch(batch_size).prefetch(10*batch_size)
    iterz = iterz.make_one_shot_iterator()

    # Return the iterator for training.
    return iterz.get_next()
'''
    # Test
    namez = iterz.get_next()
    with tf.Session() as session:
        while True:
            namez1 = session.run(namez)
            print(namez1[2])
            print(namez1[1])
            ima = namez1[0]
            aud = namez1[3]
            signal = aud[0, :, 0]
            print(signal)
            plt.figure(1)
            plt.title('Signal Wave...')
            plt.plot(signal)
            plt.show()
            fig = pylab.figure()
            fig.suptitle('image #{}'.format(1), fontsize=20)
            pylab.imshow(ima[0, :, :, 0, 0])
'''


def load_test(fpsTest, batch_sizeTest):
    # Create a dataset of filepaths.
    datasetTest = tf.data.Dataset.from_tensor_slices(fpsTest)

    # Preprocess audio and video frames, and create a dataset of labels.
    audio_datasetTest = datasetTest.map(decode_audio)
    label_datasetTest = datasetTest.map(decode_video_labels)
    frame_datasetTest = datasetTest.map(decode_video_frames)

    # Connect all three datasets together for batching datasets in the same order.
    dataset_zippedTest = tf.data.Dataset.zip((frame_datasetTest, label_datasetTest,
                                              datasetTest, audio_datasetTest))

    # Shuffle dataset for better training.
    # Shuffling takes too long. So, for testing, use the second line that doesn't
    # shuffle the datasets. For real training, use the first line.
    dataset_zippedTest = dataset_zippedTest.shuffle(buffer_size=len(fpsTest))\
        .repeat().shuffle(buffer_size=len(fpsTest))
    #dataset_zippedTest = dataset_zippedTest.repeat()

    # Create an iterator for pipelining the dataset into the model.
    iterzTest = dataset_zippedTest.batch(batch_sizeTest).prefetch(10*batch_sizeTest)
    iterzTest = iterzTest.make_one_shot_iterator()

    # Return the iterator for training.
    return iterzTest.get_next()

def train(fps_t, args_t, fps_test, args_test):
    #with tf.device("/gpu:0"):
    #with tf.name_scope('loader'):
    x = load_batch(fps_t, args_t.batch_size)
    #x_audio_b = x[3]
    x_frames_b = x[0]
    labels_b = x[1]

    x_test = load_test(fps_test, args_test.batch_size)
    #x_audio_test = x_test[3]
    x_frames_test = x_test[0]
    labels_test = x_test[1]

    x_frames = tf.placeholder("float", [None, 64, 64, 16, 1])
    #x_audio = tf.placeholder("float", [None, 131072, 1])
    labels = tf.placeholder("int64", [None])

    y = FrameClassifier64(x_frames, use_batchnorm=True, train=True)
    #y_audio = AudioClassifier(x_audio, use_batchnorm=True, train=True)

    #y = tf.concat([y_frame, y_audio], axis=1)

    out = OutputClassifier(y, use_batchnorm=True, train=True)

    # Create loss, set optimizer and initialize variables
    learning_rate = 0.00001
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits
                          (logits=out, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Print messages
    correct_prediction = tf.equal(tf.argmax(out, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(args_t.sim_iter):
            for ii in range(10):
                ima_t, lab_t, name_t = sess.run([x_frames_b, labels_b, x[2]])

                # Optimization-Training
                opt = sess.run(optimizer, feed_dict={x_frames: ima_t,
                                                     labels: lab_t})
                loss, acc = sess.run([cost, accuracy], feed_dict={x_frames: ima_t,
                                                                  labels: lab_t})
                print("Epoch " + str(i) + \
                      " Iter " + str(ii) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

            test_acc = 0
            valid_loss = 0
            for iii in range(20):
                # Calculate testing-accuracy for all test videos.
                ima_test, lab_test, name_test = sess.run(
                    [x_frames_test, labels_test, x_test[2]])
                acc_t, loss_v = sess.run([accuracy, cost],
                                         feed_dict={x_frames: ima_test,
                                                    labels: lab_test})
                test_acc = test_acc + acc_t
                valid_loss = valid_loss + loss_v

            test_acc = test_acc/20
            valid_loss = valid_loss/20

            train_loss.append(loss)
            test_loss.append(valid_loss)
            train_accuracy.append(acc)
            test_accuracy.append(test_acc)
            print("Testing Accuracy:", "{:.5f}".format(test_acc))

        summary_writer.close()

        # Loss plots
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
        plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
        plt.title('Training and Test loss')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.show()

        # Accuracy plots
        plt.figure()
        plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
        plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
        plt.title('Training and Test Accuracy')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.show()


# Main function, which runs the following operations:
    # 1. Prepare data (import video, split it into frames and audio),
    # 2. Train the neural network,
    # 3. Test the accuracy of neural network./home/glados/Desktop/MultimediaDNN
if __name__ == '__main__':
    #tf.enable_eager_execution()
    args = InputClass()

    setattr(args, 'data_dir', '../../../../media/glados/New Volume/Linux/'
                              'Asilomar/Data1/train')
    setattr(args, 'batch_size', 12)
    setattr(args, 'latent_dim', 200)
    setattr(args, 'sim_iter', 50)


    filepaths = glob.glob(os.path.join(args.data_dir, '*/*'))
    print('Found {} audio files in specified directory'.format(len(filepaths)))

    test_args = InputClass()

    setattr(test_args, 'data_dir', '../../../../media/glados/New Volume/Linux/'
                              'Asilomar/Data1/test')
    setattr(test_args, 'batch_size', 8)
    setattr(test_args, 'latent_dim', 200)
    test_file_num = 32
    setattr(test_args, 'sim_iter', test_file_num/test_args.batch_size)

    test_filepaths = glob.glob(os.path.join(test_args.data_dir, '*/*'))
    print('Found {} audio files in specified directory'.format(len(test_filepaths)))

    train(filepaths, args, test_filepaths, test_args)