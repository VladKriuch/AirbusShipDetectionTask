import keras.backend as K
import numpy as np
import skimage

"""
Script for estimating overall dice coef
"""

# constants
BATCH_SIZE = 4
IMG_FOLDER_PATH = '../train_images/'


def dice_coef(y_true, y_pred, smooth=1.):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = y_true_f.astype(np.int32)
    y_pred_f = y_pred_f.astype(np.int32)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
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
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def get_mask(img_id, df):
  img = masks_as_image(df.query('ImageId=="'+img_id+'"')['EncodedPixels'])
  img = skimage.transform.resize(img, (384, 384), mode='constant', preserve_range=True)
  return img


def get_image(image_path):
    img = skimage.io.imread(IMG_FOLDER_PATH + image_path)
    img = skimage.transform.resize(img, (384, 384))
    return skimage.img_as_uint(img)


def make_image_gen(in_df, batch_size=BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    masks = []
    images = []
    while True:
        np.random.shuffle(all_batches)
        for image, masks_df in all_batches:
            image, mask = [get_image(image)], [get_mask(image, in_df)]
            images += image
            masks += mask
            if len(images)>=batch_size:
                yield np.stack(images, 0), np.stack(masks, 0)
                masks, images=[], []


if __name__ == "__main__":
    from model_inference import SegmentationModel
    from model_structures import vgg16_classifier
    from model_structures import unet_segmentator

    classifier_model = vgg16_classifier.get_model()
    segmentator_model = unet_segmentator.get_model()
    segmentation_model = \
        SegmentationModel(classifier=classifier_model, classifier_weigths='../model_weights/CLASS 0.9.hdf5',
                          segmentator=segmentator_model, segmentation_weights='../model_weights/DICE 0.7.hdf5')

    import pandas as pd

    train_df = pd.read_csv('../helpers/train_file.csv')

    dice_coefs = []
    img_mask_gen = make_image_gen(train_df)
    count = 0
    for _ in range(32):
        count += 1
        print(f'Step {count}')
        # Around 128 igms
        images, masks = next(img_mask_gen)
        predictions = segmentation_model.predict(images)
        dice_coef_local = 0
        for indx, mask in enumerate(masks):
            dice_coef_local += dice_coef(mask, predictions[indx])
        dice_coef_local /= len(masks)
        dice_coefs.append(dice_coef_local)

        del images
        del masks
        del predictions
    print(dice_coefs)
    print(sum(dice_coefs) / len(dice_coefs))