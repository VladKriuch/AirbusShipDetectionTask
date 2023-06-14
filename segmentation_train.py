import pandas as pd
import numpy as np
import keras.backend as K

from skimage.transform import resize
from skimage.io import imread
from skimage import img_as_ubyte

from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# constants
DATA_PATH = ''
TRAIN_PATH = DATA_PATH+'train_images/'
TEST_PATH = DATA_PATH+'train_images/'
IMG_SIZE = (768, 768)
INPUT_SHAPE = (768, 768, 3)
TARGET_SIZE = (384, 384)
TARGET_SIZE_RGB = (384, 384, 3)
BATCH_SIZE = 48


def rle_encode(img):
    """
    Method for encoding rle mask
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    """
    Method for decoding rle mask
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list, all_masks=None):
    # Take the individual ship masks and create a single mask array for all ships
    if all_masks is None:
        all_masks = np.zeros((768, 768), dtype = np.int16)

    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)

    return np.expand_dims(all_masks, -1)


def get_mask(img_id, df):
    """
    Get mask for single image
    :param img_id: image id - file name
    :param df: dataframe where images and masks are stored
    """
    img = masks_as_image(df.query('ImageId=="'+img_id+'"')['EncodedPixels'])
    img = resize(img, TARGET_SIZE, mode='constant', preserve_range=True)
    return img


def get_image(image_name):
    """
    Get image
    """
    img = imread(TRAIN_PATH + image_name)
    img = resize(img, (384, 384))
    return img_as_ubyte(img)


def make_image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    masks = []
    images = []
    while True:
        np.random.shuffle(all_batches)
        for image, masks_df in all_batches:
            image, mask = [get_image(image)], [get_mask(image, masks_df)]
            images += image
            masks += mask

            if len(images)>=batch_size:
                yield np.stack(images, 0)/255.0, np.stack(masks, 0)
                masks, images=[], []


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = y_true_f.astype(np.float32)
    y_pred_f = y_pred_f.astype(np.float32)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Model related imports
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, \
    TerminateOnNaN, TensorBoard
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, Dropout


def get_unet_model(input_shape=(256, 256, 3), num_classes=1):
    """
    Create and get unet-shaped model
    """
    def fire(x, filters, kernel_size, dropout):
        y1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
        if dropout is not None:
            y1 = Dropout(dropout)(y1)
        y2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(y1)
        y3 = BatchNormalization(momentum=0.95)(y2)
        return y3

    def fire_module(filters, kernel_size, dropout=0.1):
        return lambda x: fire(x, filters, kernel_size, dropout)

    def fire_up(x, filters, kernel_size, concat_layer):
        y1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        y2 = concatenate([y1, concat_layer])
        y3 = fire_module(filters, kernel_size, dropout=None)(y2)
        return y3

    def up_fire_module(filters, kernel_size, concat_layer):
        return lambda x: fire_up(x, filters, kernel_size, concat_layer)

    input_img = Input(shape=input_shape)  # 256

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

    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(up10)

    model = Model(inputs=[input_img], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
    return model


if __name__ == "__main__":
    # Get model
    model = get_unet_model(TARGET_SIZE_RGB)

    # Create checkpoint for saving weights
    weight_path="{}_weights.best.hdf5".format('model')
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    callbacks_list = [checkpoint]

    # Read train file; Prepare train data
    train_df = pd.read_csv('helpers/train_file.csv')
    train_df = train_df.dropna()
    unique_masks = train_df.drop_duplicates('ImageId')

    train_df_l, validate_df_l = train_test_split(unique_masks) # split data

    train_df_l = train_df[train_df['ImageId'].isin(train_df_l['ImageId'])]
    validate_df_l = train_df[train_df['ImageId'].isin(validate_df_l['ImageId'])]

    train_gen = make_image_gen(train_df_l)
    validation_gen = next(make_image_gen(validate_df_l)) # Not using generator because of resources

    step_count = 25
    history = model.fit(train_gen,
                                 steps_per_epoch=step_count,
                                 epochs=100,
                                 validation_data=validation_gen,
                                 callbacks=callbacks_list)
