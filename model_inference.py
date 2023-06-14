import skimage
import numpy as np


class SegmentationModel:
    """
    Whole segmantation model
    Combined from Classification model and Segmentation model
    """
    def __init__(self, classifier, segmentator, classifier_weigths: str = "", segmentation_weights: str = ""):
        """
        :param classifier: Classifier model, gets Image, returns 1 if ships on image, 0 if ships are not presented
        :param segmentator: Segmantation model, gets Image, returns mask of this image

        :param classifier_weigths: Weights file for Classifier model
        :param segmentation_weights: Weights file for segmentation model
        """
        if classifier_weigths:
            classifier.load_weights(classifier_weigths)

        if segmentation_weights:
            segmentator.load_weights(segmentation_weights)

        self.classifier = classifier
        self.segmentator = segmentator

    def predict(self, X, print_classes=False, *args, **kwargs):
        """
        :param X: Input array of images
        :param print_classes: Print classes after classificatin model is done
        :return: Output - masks for input images
        """
        if X.shape != (X.shape[0], 384, 384, 3):
            # Resize images if they are not required shape
            X = skimage.img_as_uint(skimage.transform.resize(X, (X.shape[0], 384, 384, 3)))

        classifier_predictions = self.classifier.predict(X / 255., *args, **kwargs)
        classes = np.argmax(classifier_predictions, axis=1)  # Predict if there is ship on image
        if print_classes:
            print(classes)

        output = []
        for indx, result in enumerate(classes):
            if result == 0:
                # Found no ship, means we do not need to find mask for image
                output.append(np.zeros((384, 384, 1)))
            else:
                # Found ship, predicting its mask
                output.append(self.segmentator.predict(skimage.img_as_ubyte(np.array([X[indx]])) / 255.,
                                                       *args, **kwargs)[0])

        return np.array(output).reshape((len(output), 384, 384, 1))

    @staticmethod
    def get_image(image_path):
        """
        Gets image placed in image_path variable
        :param image_path: path to image
        :return:
        """
        img = skimage.io.imread(image_path)
        # img = skimage.transform.resize(img, (384, 384))
        return skimage.img_as_ubyte(img)


if __name__ == "__main__":
    # Import model structures
    from model_structures import vgg16_classifier
    from model_structures import unet_segmentator

    classifier_model = vgg16_classifier.get_model()
    segmentator_model = unet_segmentator.get_model()
    segmentation_model = \
        SegmentationModel(classifier=classifier_model, classifier_weigths='model_weights/CLASS 0.9.hdf5',
                          segmentator=segmentator_model, segmentation_weights='model_weights/DICE 0.7.hdf5')

    # Get image examples
    image = segmentation_model.get_image('train_images/0a0a623a0.jpg')
    image2 = segmentation_model.get_image('train_images/00a52cd2a.jpg')
    image3 = segmentation_model.get_image('train_images/00bc708e0.jpg')
    image4 = segmentation_model.get_image('train_images/00c4be6fa.jpg')

    # Predict masks
    masks = segmentation_model.predict(np.array([image, image2, image3, image4]), print_classes=True)

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(4, 2)
    # Good prediction NO SHIP
    axs[0, 0].imshow(image)
    axs[0, 1].imshow(masks[0])

    # Good segmentation SHIP
    axs[1, 0].imshow(image2)
    axs[1, 1].imshow(masks[1])

    # Medium quality segmantation SHIP ( small size )
    axs[2, 0].imshow(image3)
    axs[2, 1].imshow(masks[2])

    # Good prediction NO SHIP with island
    axs[3, 0].imshow(image4)
    axs[3, 1].imshow(masks[3])

    plt.show()

