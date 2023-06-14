import keras
import pandas as pd
import numpy as np

from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_ubyte

from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16

# constants
DATA_PATH = ''
TRAIN_PATH = DATA_PATH+'train_images/'
TEST_PATH = DATA_PATH+'train_images/'
IMG_SIZE = (768, 768)
INPUT_SHAPE = (768, 768, 3)
TARGET_SIZE = (384, 384)
TARGET_SIZE_RGB = (384, 384, 3)
BATCH_SIZE = 4
EPOCHS = 100


def get_image(image_name):
    img = imread(TRAIN_PATH + image_name)
    img = resize(img, (384, 384))
    return img_as_ubyte(img)


def make_image_gen(X, Y, batch_size=BATCH_SIZE):
    labels = []
    images = []
    while True:
        for indx in range(len(X)):
            image, label = [get_image(X[indx])], [Y[indx]]

            images += image
            labels += label

            if len(images)>=batch_size:

                yield np.stack(images, 0), to_categorical(np.stack(labels, 0))
                labels, images=[], []


if __name__ == "__main__":
    # Read file with data
    train_df = pd.read_csv('helpers/train_file.csv')

    # Prepare data for classification task
    train_df = train_df.drop_duplicates('ImageId')
    train_df['IsEmpty'] = train_df['EncodedPixels'].notna()

    from sklearn.model_selection import train_test_split

    train_img_ids, test_img_ids, y_train, y_test = train_test_split(train_df['ImageId'], train_df['IsEmpty'],
                                                                    test_size=0.3, random_state=42)
    train_gen = make_image_gen(train_img_ids.to_numpy(), y_train.to_numpy(), 32)
    validation_gen = next(make_image_gen(test_img_ids.to_numpy(), y_test.to_numpy(), 32))

    # create model
    base_model = VGG16(input_shape=(384, 384, 3), weights='imagenet', include_top=False)
    base_model.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(2, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # create checkpoint
    from keras.callbacks import ModelCheckpoint

    weight_path = "{}_weights.best.hdf5".format('model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                 save_weights_only=True)
    step_count = 25
    history = model.fit(train_gen,
                                 steps_per_epoch=step_count,
                                 epochs=EPOCHS,
                                 validation_data=validation_gen,
                                 callbacks=[checkpoint])
