import keras
from tensorflow.keras.applications.vgg16 import VGG16

INPUT_SHAPE = (384, 384, 3)


def get_model():
    base_model = VGG16(input_shape=(384, 384, 3), weights='imagenet', include_top=False)
    base_model.trainable = False
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(2, activation='softmax')(x)

    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    return model