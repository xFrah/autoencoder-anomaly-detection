# USAGE
# python train_unsupervised_autoencoder.py --dataset output/images.pickle --model output/autoencoder.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
from keras.api._v2.keras.layers.experimental import preprocessing
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential

matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from PIL import Image
import pickle
import cv2
import os
import imageio


def build_unsupervised_dataset(data, labels, validLabel=0,
                               anomalyLabel=1, contam=0.01, seed=42):
    # grab all indexes of the supplied class label that are *truly*
    # that particular label, then grab the indexes of the image
    # labels that will serve as our "anomalies"
    validIdxs = np.where(labels == validLabel)[0]
    anomalyIdxs = np.where(labels == anomalyLabel)[0]

    # randomly shuffle both sets of indexes
    random.shuffle(validIdxs)
    random.shuffle(anomalyIdxs)

    # compute the total number of anomaly data points to select
    i = int(len(validIdxs) * contam)
    anomalyIdxs = anomalyIdxs[:i]

    # use NumPy array indexing to extract both the valid images and
    # "anomlay" images
    validImages = data[validIdxs]
    anomalyImages = data[anomalyIdxs]

    # stack the valid images and anomaly images together to form a
    # single data matrix and then shuffle the rows
    images = np.vstack([validImages, anomalyImages])
    np.random.seed(seed)
    np.random.shuffle(images)

    # return the set of images
    return images


def visualize_predictions(decoded, gt, samples=10):
    # initialize our list of output images
    outputs = None

    # loop over our number of output samples
    for i in range(0, len(decoded)):
        # grab the original image and reconstructed image
        original = (gt[i] * 255).astype("uint8")
        recon = (decoded[i] * 255).astype("uint8")

        # stack the original and reconstructed image side-by-side
        output = np.hstack([original, recon])

        # if the outputs array is empty, initialize it as the current
        # side-by-side image display
        if outputs is None:
            outputs = output

        # otherwise, vertically stack the outputs
        else:
            outputs = np.vstack([outputs, output])

    # return the output images
    return outputs


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
                help="path to output dataset file")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to output trained autoencoder")
ap.add_argument("-v", "--vis", type=str, default="recon_vis.png",
                help="path to output reconstruction visualization file")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output plot file")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 1000
INIT_LR = 1e-3
BS = 14

paper_dir, other_dir = os.listdir("paper"), os.listdir("other")
trainX = [imageio.imread("paper/" + f) for f in paper_dir] + [imageio.imread("other/" + f) for f in other_dir]
trainX = np.array([cv2.resize(i, (45, 45)) for i in trainX])
trainY = np.array(([0] * len(paper_dir)) + ([1] * len(other_dir)))

# print(trainX.shape)

# trainX: np.ndarray

# print("[INFO] loading MNIST dataset...")
# ((trainX, trainY), (testX, testY)) = mnist.load_data()

# print(trainX.shape)

# build our unsupervised dataset of images with a small amount of
# contamination (i.e., anomalies) added into it
print("[INFO] creating unsupervised dataset...")
images = build_unsupervised_dataset(trainX, trainY, contam=0.0)
images2 = build_unsupervised_dataset(trainX, trainY, contam=0.07)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)  <<1    0 => 30
    shear_range=0.2,  # set range for random shear  <<3<<4  0 => 0.1 => 0.2
    zoom_range=0.3,  # set range for random zoom    <<1<<2<<3   0 => 0.1 => 0.2 =>0.3
    channel_shift_range=0.2,  # set range for random channel shifts     <<5<<6   0.=>0.1=>0.2
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images    <<1    false => True
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format="channels_last",
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
# images = np.expand_dims(images, axis=-1)
images = images.astype("float32") / 255.0
images2 = images2.astype("float32") / 255.0

# construct the training and testing split
(trainX, testX) = train_test_split(images, test_size=0.2,
                                   random_state=42)

# construct the training and testing split
(trainX2, testX2) = train_test_split(images2, test_size=0.2,
                                     random_state=42)

# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
(encoder, decoder, autoencoder) = ConvAutoencoder.build(45, 45, 3)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)

# train the convolutional autoencoder
H = autoencoder.fit(
    datagen.flow(trainX, trainX,
                 batch_size=BS),
    steps_per_epoch=2,
    validation_data=datagen.flow(testX2, testX2,
                                 batch_size=BS),
    epochs=EPOCHS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX2)
vis = visualize_predictions(decoded, testX2)
cv2.imwrite(args["vis"], vis)

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the image data to disk
print("[INFO] saving image data...")
f = open(args["dataset"], "wb")
f.write(pickle.dumps(images2))
f.close()

# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save(args["model"], save_format="h5")
