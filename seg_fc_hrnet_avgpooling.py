import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import UpSampling2D, add, concatenate, Dropout, AveragePooling2D


def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def _make_transition_layer(x, out_filters_list=[32, 64, 96, 128, 160, 192, 224, 256]):
    transition_layers = []
    xi = Conv2D(out_filters_list[0], 3, strides=(1, 1),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    xi = BatchNormalization(axis=3)(xi)
    xi = Activation('relu')(xi)
    transition_layers.append(xi)

    for i in range(1, len(out_filters_list)):
        xi = Conv2D(out_filters_list[i], 3, strides=(2, 2),
                    padding='same', use_bias=False, kernel_initializer='he_normal')(xi)
        xi = BatchNormalization(axis=3)(xi)
        xi = Activation('relu')(xi)
        transition_layers.append(xi)

    return transition_layers


def make_branch(x, out_filters=32):
    x = basic_Block(x, out_filters, with_conv_shortcut=False)
    # x = basic_Block(x, out_filters, with_conv_shortcut=False)
    # x = basic_Block(x, out_filters, with_conv_shortcut=False)
    # x = basic_Block(x, out_filters, with_conv_shortcut=False)
    return x


def _make_fuse_layers(x, num_branches, out_filters_list=[32, 64, 96, 128, 160, 192, 224, 256], multi_scale_output=False):
    fuse_layers = []
    for i in range(num_branches if multi_scale_output else 1):
        fuse_layer = []
        for j in range(num_branches):
            if j == i:
                fuse_layer.append(x[i])
            elif j > i:
                xi = Conv2D(out_filters_list[i], 1, strides=(1, 1),
                            padding='same', use_bias=False, kernel_initializer='he_normal')(x[j])
                xi = BatchNormalization(axis=3)(xi)
                xi = UpSampling2D(size=(2**(j-i), 2**(j-i)))(xi)
                fuse_layer.append(xi)
            elif j < i:
                xi = Conv2D(out_filters_list[i], 1, use_bias=False, kernel_initializer='he_normal')(x[j])
                xi = AveragePooling2D(pool_size=(2**(i-j), 2**(i-j)), strides=(2**(i-j), 2**(i-j)))(xi)
                xi = BatchNormalization(axis=3)(xi)
                fuse_layer.append(xi)
        xi = add(fuse_layer)
        fuse_layers.append(xi)
    return fuse_layers


def seg_fc_hrnet(height=512, width=512, channel=3, classes=6):
    inputs = Input(shape=(height, width, channel))

    out_filters_list = [32, 64, 96, 128, 160, 192, 224, 256]
    x = _make_transition_layer(inputs, out_filters_list=out_filters_list)

    x1 = []
    for i in range(len(out_filters_list)):
        x1.append(make_branch(x[i], out_filters=out_filters_list[i]))

    x1 = _make_fuse_layers(x1, len(out_filters_list),
                           out_filters_list=out_filters_list,
                           multi_scale_output=True)

    x2 = []
    for i in range(len(out_filters_list)):
        x2.append(make_branch(x1[i], out_filters=out_filters_list[i]))

    x2 = _make_fuse_layers(x2, len(out_filters_list),
                           out_filters_list=out_filters_list,
                           multi_scale_output=True)

    x3 = []
    for i in range(len(out_filters_list)):
        x3.append(make_branch(x2[i], out_filters=out_filters_list[i]))

    x3 = _make_fuse_layers(x3, len(out_filters_list),
                           out_filters_list=out_filters_list,
                           multi_scale_output=False)

    out = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x3[0])
    out = BatchNormalization(axis=3)(out)
    out = Activation('softmax', name='Classification')(out)

    model = Model(inputs=inputs, outputs=out)

    return model


model = seg_fc_hrnet(height=512, width=512, channel=3, classes=6)
model.summary()

from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='seg_fc_hrnet.png', show_shapes=True, show_layer_names=True)