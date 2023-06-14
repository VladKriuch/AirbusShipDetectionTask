from tensorflow.python.ops.numpy_ops import np_config
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose
from keras import Model

np_config.enable_numpy_behavior()

INPUT_SHAPE = (384, 384, 3)
NUM_CLASSES = 1


def get_model():
    def fire(x, filters, kernel_size, dropout):
        y1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        y2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(y1)
        y3 = BatchNormalization(momentum=0.95)(y2)
        return y3

    def fire_module(filters, kernel_size, dropout=0.1):
        return lambda x: fire(x, filters, kernel_size, dropout)

    def fire_up(x, filters, kernel_size, concat_layer):
        y1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        y2 = concatenate([y1, concat_layer])
        y3 = fire_module(filters, kernel_size)(y2)
        return y3

    def up_fire_module(filters, kernel_size, concat_layer):
        return lambda x: fire_up(x, filters, kernel_size, concat_layer)

    input_img = Input(shape=INPUT_SHAPE)  # 256

    down1 = fire_module(8, (3, 3))(input_img)
    pool1 = MaxPooling2D((2, 2))(down1)  # 128

    down2 = fire_module(16, (3, 3))(pool1)
    pool2 = MaxPooling2D((2, 2))(down2)  # 64

    down3 = fire_module(32, (3, 3))(pool2)
    pool3 = MaxPooling2D((2, 2))(down3)  # 32

    down4 = fire_module(64, (3, 3))(pool3)
    pool4 = MaxPooling2D((2, 2))(down4)  # 16

    down5 = fire_module(128, (3, 3))(pool4)
    pool5 = MaxPooling2D((2, 2))(down5)  # 8

    down6 = fire_module(256, (3, 3))(pool5)  # center

    up6 = up_fire_module(128, (3, 3), down5)(down6)  # 16
    up7 = up_fire_module(64, (3, 3), down4)(up6)  # 32
    up8 = up_fire_module(32, (3, 3), down3)(up7)  # 64
    up9 = up_fire_module(16, (3, 3), down2)(up8)  # 128
    up10 = up_fire_module(8, (3, 3), down1)(up9)  # 256

    outputs = Conv2D(NUM_CLASSES, (1, 1), activation='sigmoid')(up10)

    model = Model(inputs=[input_img], outputs=[outputs])

    return model