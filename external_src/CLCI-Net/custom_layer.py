# -*- coding: utf-8 -*-
from keras import *
from keras.layers import *
import tensorflow as tf
kernel_regularizer = regularizers.l2(1e-5)
bias_regularizer = regularizers.l2(1e-5)
kernel_regularizer = None
bias_regularizer = None

def conv_lstm(input1, input2, channel=256):
    lstm_input1 = Reshape((1, input1.shape.as_list()[1], input1.shape.as_list()[2], input1.shape.as_list()[3]))(input1)
    lstm_input2 = Reshape((1, input2.shape.as_list()[1], input2.shape.as_list()[2], input1.shape.as_list()[3]))(input2)

    lstm_input = custom_concat(axis=1)([lstm_input1, lstm_input2])
    x = ConvLSTM2D(channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(lstm_input)
    return x

def conv_2(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(conv_)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_2_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)

def conv_2_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def conv_1(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = kernel_regularizer):
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_1_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)

def conv_1_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4))

def dilate_conv(inputs, filter_num, dilation_rate):
    conv_ = Conv2D(filter_num, kernel_size=(3,3), dilation_rate=dilation_rate, padding='same', kernel_initializer='he_normal', kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation('relu')(conv_)
    return conv_

class custom_concat(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(custom_concat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.built = True
        super(custom_concat, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.res = tf.concat(x, self.axis)

        return self.res

    def compute_output_shape(self, input_shape):
        # return (input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:]
        # print((input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:])
        input_shapes = input_shape
        output_shape = list(input_shapes[0])

        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]

        return tuple(output_shape)


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.upsampling = upsampling

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))



def concat_pool(conv, pool, filter_num, strides=(2, 2)):
    conv_downsample = Conv2D(filter_num, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(conv)
    conv_downsample = BatchNormalization()(conv_downsample)
    conv_downsample = Activation('relu')(conv_downsample)
    concat_pool_ = Concatenate()([conv_downsample, pool])
    return concat_pool_

