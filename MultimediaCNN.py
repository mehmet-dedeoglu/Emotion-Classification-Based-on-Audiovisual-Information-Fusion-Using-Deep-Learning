import tensorflow as tf


def conv1d_transpose(inputs, filters, kernel_width, stride=4,
                     padding='same', upsample='zeros'):


    if upsample == 'zeros':
        return tf.layers.conv2d_transpose(
            tf.expand_dims(inputs, axis=1),
            filters,
            (1, kernel_width),
            strides=(1, stride),
            padding='same'
            )[:, 0]
    elif upsample == 'nn':
        batch_size = tf.shape(inputs)[0]
        _, w, nch = inputs.get_shape().as_list()

        x = inputs

        x = tf.expand_dims(x, axis=1)
        x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
        x = x[:, 0]

        return tf.layers.conv1d(
            x,
            filters,
            kernel_width,
            1,
            padding='same')
    else:
        raise NotImplementedError


def lrelu(inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])
    
    return x


def AudioClassifier(z, kernel_len=25, dim=64, use_batchnorm=False,
                         phaseshuffle_rad=0, train=False):
    batch_size = tf.shape(z)[0]
    slice_len = int(z.get_shape()[1])

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    if phaseshuffle_rad > 0:
        phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
        phaseshuffle = lambda x: x

    # Layer 0
    # [131072, 1] -> [16384, 64]
    output = z
    with tf.variable_scope('downconvaudio_0'):
        output = tf.layers.conv1d(output, dim, kernel_len, 8, padding='SAME')
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # 16384, 64] -> [2048, 128]
    with tf.variable_scope('downconvaudio_1'):
        output = tf.layers.conv1d(output, dim * 2, kernel_len, 8, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [2048, 128] -> [256, 256]
    with tf.variable_scope('downconvaudio_2'):
        output = tf.layers.conv1d(output, dim * 4, kernel_len, 8, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconvaudio_3'):
        output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconvaudio_4'):
        output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, dim * 16 * 16])


    # Can be removed!
    # Connect to single logit
    #with tf.variable_scope('output'):
    #    output = tf.layers.dense(output, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the generator
    # because we only use the discriminator for training
    #if train and use_batchnorm:
    #    update_ops2 = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
    #                                    scope=tf.get_variable_scope().name)
    #    if slice_len == 16384:
    #        assert len(update_ops2) == 10
    #    else:
    #        assert len(update_ops2) == 12
    #    with tf.control_dependencies(update_ops2):
    #        output = tf.identity(output)

    return output


def FrameClassifier(x, kernel_len=3, dim=64, use_batchnorm=False, train=False):
    batch_size = tf.shape(x)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # Layer 0
    # [128, 128, 16, 1] -> [64, 64, 16, 64]
    output = x
    with tf.variable_scope('downconv_0'):
        output = tf.layers.conv3d(output, dim, kernel_len, [2, 2, 1],
                                  padding='SAME')
    output = lrelu(output)

    # Layer 1
    # [64, 64, 16, 64] -> [32, 32, 16, 128]
    with tf.variable_scope('downconv_1'):
        output = tf.layers.conv3d(output, dim * 2, kernel_len, [2, 2, 1],
                                  padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 2
    # [32, 32, 16, 128] -> [16, 16, 16, 256]
    with tf.variable_scope('downconv_2'):
        output = tf.layers.conv3d(output, dim * 4, kernel_len, [2, 2, 1],
                                  padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 3
    # [16, 16, 16, 256] -> [8, 8, 8, 512]
    with tf.variable_scope('downconv_3'):
        output = tf.layers.conv3d(output, dim * 8, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 4
    # [8, 8, 8, 512] -> [4, 4, 4, 1024]
    with tf.variable_scope('downconv_4'):
        output = tf.layers.conv3d(output, dim * 16, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 5
    # [4, 4, 4, 1024] -> [2, 2, 2, 2048]
    with tf.variable_scope('downconv_5'):
        output = tf.layers.conv3d(output, dim * 32, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, 2 * 2 * 2 * dim * 32])

    # Can be removed!
    # Connect to single logit
    #with tf.variable_scope('output'):
    #    output = tf.layers.dense(output, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the
    # generator because we only use the discriminator for training
    return output

def FrameClassifier64(x, kernel_len=3, dim=128, use_batchnorm=False, train=False):
    batch_size = tf.shape(x)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    # Layer 0
    # [64, 64, 16, 1] -> [32, 32, 16, 128]
    output = x
    with tf.variable_scope('downconv_0'):
        output = tf.layers.conv3d(output, dim, kernel_len, [2, 2, 1],
                                  padding='SAME')
    output = lrelu(output)

    # Layer 1
    # [32, 32, 16, 128] -> [16, 16, 16, 256]
    with tf.variable_scope('downconv_1'):
        output = tf.layers.conv3d(output, dim * 2, kernel_len, [2, 2, 1],
                                  padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 3
    # [16, 16, 16, 256] -> [8, 8, 8, 512]
    with tf.variable_scope('downconv_3'):
        output = tf.layers.conv3d(output, dim * 4, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 4
    # [8, 8, 8, 512] -> [4, 4, 4, 1024]
    with tf.variable_scope('downconv_4'):
        output = tf.layers.conv3d(output, dim * 8, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Layer 5
    # [4, 4, 4, 1024] -> [2, 2, 2, 2048]
    with tf.variable_scope('downconv_5'):
        output = tf.layers.conv3d(output, dim * 16, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)

    # Flatten
    output = tf.reshape(output, [batch_size, 2 * 2 * 2 * dim * 16])

    # Can be removed!
    # Connect to single logit
    #with tf.variable_scope('output'):
    #    output = tf.layers.dense(output, 1)[:, 0]

    # Don't need to aggregate batchnorm update ops like we do for the
    # generator because we only use the discriminator for training
    return output


def OutputClassifier(_input, use_batchnorm=False, train=True):
    batch_size = tf.shape(_input)[0]

    if use_batchnorm:
        batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
        batchnorm = lambda x: x

    output = _input
    with tf.variable_scope('output'):
        # 2 if class number is 2 and 8 if class number is 8.
        output = tf.layers.dense(output, 2)
        output = batchnorm(output)
    #    output = tf.contrib.layers.softmax(output)[:, 0]

    return output