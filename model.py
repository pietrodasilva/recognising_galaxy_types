###################### - BOILER PLATE CODE - ###################################

import numpy as np
import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, \
    Resizing, CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, \
    RandomRotation, RandomZoom
from tensorflow.python.keras.preprocessing.image_dataset import \
    image_dataset_from_directory

################## - PROCESSING TRAINING IMAGES - ##############################

# training_images = []
# training_labels = []
shape_size = (424, 424)
batch_size = 32
num_folds = 10

img_height = 424
img_width = 424

training_path = 'C:/Users/PietroPC/' \
                'OneDrive - University of Essex/University/Year 3/' \
                'CE301 Capstone Final Project/' \
                'git/ce301_carneiro_da_silva_pietro_e/' \
                'galaxy_project/image_database/training_set'

testing_path = 'C:/Users/PietroPC/' \
               'OneDrive - University of Essex/University/Year 3/' \
               'CE301 Capstone Final Project/' \
               'git/ce301_carneiro_da_silva_pietro_e/' \
               'galaxy_project/image_database/testing_set'

dir_path = "image_database/training_set"

data_dir = pathlib.Path(dir_path)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

list_dataset = tf.data.Dataset.list_files(str(data_dir / '*/*'), shuffle=False)
list_dataset = list_dataset.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_dataset.take(5):
    print(f.numpy())

class_names = np.array(sorted(
    [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

training_data = image_dataset_from_directory(training_path,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=123,
                                             label_mode='categorical',
                                             color_mode='grayscale',
                                             shuffle=True,
                                             batch_size=batch_size,
                                             image_size=shape_size)

validation_data = image_dataset_from_directory(training_path,
                                               validation_split=0.2,
                                               subset='validation',
                                               seed=123,
                                               label_mode='categorical',
                                               color_mode='grayscale',
                                               shuffle=False,
                                               batch_size=batch_size,
                                               image_size=shape_size)

test_data = image_dataset_from_directory(testing_path,
                                         label_mode='categorical',
                                         color_mode='grayscale',
                                         shuffle=True,
                                         batch_size=batch_size,
                                         image_size=shape_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE

training_dataset = training_data.prefetch(buffer_size=AUTOTUNE)
validation_dataset = training_data.prefetch(buffer_size=AUTOTUNE)
testing_dataset = validation_data.prefetch(buffer_size=AUTOTUNE)


################################################################################

################# - DATA AUGMENTATION & PREPROCESSING - ########################

def data_preprocessing():
    return Sequential(
        [
            Rescaling(1.0 / 255),
            CenterCrop(212, 212),
            Resizing(64, 64),
        ]
    )


def data_augmentation():
    return Sequential(
        [
            RandomFlip('horizontal_and_vertical',
                       input_shape=(shape_size[0], shape_size[0], 1)),
            RandomRotation(0.2),
            RandomZoom(0.1),
        ]
    )


################################################################################

################## - MODEL CHECKPOINTS & EARLY STOPPING - ######################

# Adds a path for where the checkpoint will be saved, specifying that it
# only saves the epochs
checkpoint_path = 'C:/Users/PietroPC/' \
                  'OneDrive - University of Essex/University/Year 3/' \
                  'CE301 Capstone Final Project/' \
                  'git/ce301_carneiro_da_silva_pietro_e/' \
                  'galaxy_project/' \
                  'training_checkpoints/checkpoint-{epoch:04d}.hdf5'

# Checkpoint var, which uses the ModelCheckpoint function to save each
# epoch data.
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             monitor='val_acc',
                             verbose=1,
                             save_weights_only=True,
                             save_freq='epoch')

# Early stopping will be used here to stop the model training when a monitored
# metric has stopped improving, which in this case is val_acc.
early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')


################################################################################

################ - CREATING MODELS FOR GALAXY CLASSIFICATION - #################

# Creates the galaxy model using a base Sequential model.
# Compiles with categorical_crossentropy for the loss as I'm working with
# multi-class classification. Uses the adam optimiser as this is a base standard
# for sequential models and allows me to look at the accuracy of the model
# so I can later compare with other models.
def create_galaxy_model_base():
    base_model = Sequential()
    base_model.add(data_augmentation())
    base_model.add(data_preprocessing())

    base_model.add \
        (Conv2D(16, kernel_size=3, activation='relu', input_shape=(64, 64, 1)))
    base_model.add(MaxPooling2D())

    base_model.add(Conv2D(32, kernel_size=3, activation='relu'))
    base_model.add(MaxPooling2D())

    base_model.add(Conv2D(64, kernel_size=3, activation='relu'))
    base_model.add(MaxPooling2D())

    base_model.add(Conv2D(128, kernel_size=3, activation='relu'))
    base_model.add(MaxPooling2D())
    base_model.add(Flatten())

    base_model.add(Dense(256, activation='relu'))
    base_model.add(Dropout(0.2))
    base_model.add(Dense(256, activation='relu'))
    base_model.add(Dense(5, activation='softmax'))

    adam = tf.keras.optimizers.Adam(lr=0.001)

    base_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc'],
        optimizer=adam
    )

    return base_model


# Creates the galaxy model using a VGG16 implementation.
def create_galaxy_model_vgg16():
    vgg16_model = Sequential()

    vgg16_model.add(data_augmentation())
    vgg16_model.add(data_preprocessing())

    vgg16_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                           input_shape=(64, 64, 1)))
    vgg16_model.add(MaxPooling2D(pool_size=(2, 2)))

    vgg16_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    vgg16_model.add(MaxPooling2D(pool_size=(2, 2)))
    vgg16_model.add(Flatten())

    vgg16_model.add(Dense(256, activation='relu'))
    vgg16_model.add(Dense(128, activation='relu'))
    vgg16_model.add(Dense(5, activation='softmax'))

    adam = tf.keras.optimizers.Adam(lr=0.001)

    vgg16_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc'],
        optimizer=adam
    )



    return vgg16_model


# Creates the galaxy model using a modified version of the base VGG16 CNN model,
# adding a classifier on top of the convolutional base with an added fully
# connected layer followed by a softmax layer with 5 outputs.
def create_galaxy_model_modified_vgg16():
    vgg_conv = vgg16.VGG16(weights=None,
                           include_top=False,
                           input_shape=(64, 64, 1))

    for layer in vgg_conv.layers[:]:
        layer.trainable = False

    modified_vgg16_model = Sequential()

    modified_vgg16_model.add(data_augmentation())
    modified_vgg16_model.add(data_preprocessing())

    modified_vgg16_model.add(vgg_conv)
    modified_vgg16_model.add(Flatten())

    modified_vgg16_model.add(Dense(1024, activation='relu'))
    modified_vgg16_model.add(Dropout(0.5))
    modified_vgg16_model.add(Dense(5, activation='softmax'))

    adam = tf.keras.optimizers.Adam(lr=0.001)

    modified_vgg16_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy,
        metrics=['acc'],
        optimizer=adam
    )

    return modified_vgg16_model


# A modified version of the Keras Sequential Model where I start off with 64
# filters in the Conv2D layers. For the Dense layers I also start off with an
# additional 256 filters, with a higher dropout rate.
# The main difference in this model is the optimiser used, where in the first
# iteration of this model I used the Adam optimiser, I use the RMSProp
# Optimiser here.
def create_galaxy_model_modified_sequential():
    modified_sequential = Sequential()

    modified_sequential.add(data_augmentation())
    modified_sequential.add(data_preprocessing())

    modified_sequential.add(Conv2D(64, (3, 3), activation='relu',
                                   input_shape=(64, 64, 1)))
    modified_sequential.add(MaxPooling2D(2, 2))

    modified_sequential.add(Conv2D(64, (3, 3), activation='relu'))
    modified_sequential.add(MaxPooling2D(2, 2))

    modified_sequential.add(Conv2D(128, (3, 3), activation='relu'))
    modified_sequential.add(MaxPooling2D(2, 2))

    modified_sequential.add(Conv2D(128, (3, 3), activation='relu'))
    modified_sequential.add(MaxPooling2D(2, 2))
    modified_sequential.add(Flatten())

    modified_sequential.add(Dense(512, activation='relu'))
    modified_sequential.add(Dropout(0.5))
    modified_sequential.add(Dense(512, activation='relu'))
    modified_sequential.add(Dense(5, activation='softmax'))

    rmsprop = tf.keras.optimizers.RMSprop(lr=0.001)

    modified_sequential.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['acc'],
        optimizer=rmsprop
    )

    return modified_sequential


################################################################################

################## - EVALUATING AND FITTING THE MODEL - ########################

# Evaluating the current state of the model and comparing it to the
# testing data set.
def get_model_eval(galaxy_model):
    loss, accuracy = galaxy_model.evaluate(testing_dataset,
                                           batch_size=batch_size,
                                           verbose=1)
    print('loss = ', loss)
    print('accuracy = ', accuracy)


# Fits the model to the training and validation data provided above,
# which is where I'll get to see how it performs when training on the data.
# The batch size, num of epochs and initial epoch are all set in the parameters,
# so I can modify the number of each the variables when I call this function.
def get_model_fit(galaxy_model, epochs, initial_epoch):
    history = galaxy_model.fit(training_dataset,
                               validation_data=validation_dataset,
                               batch_size=batch_size,
                               epochs=epochs,
                               initial_epoch=initial_epoch,
                               verbose=1,
                               callbacks=[]
                               )

    # history = galaxy_model.fit(training_data,
    #                            validation_data=validation_dataset,
    #                            batch_size=batch_size,
    #                            epochs=epochs,
    #                            initial_epoch=initial_epoch,
    #                            verbose=1,
    #                            callbacks=[checkpoint, early])
    return history


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=1)
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


################################################################################

################## - PLOTTING MODEL SUMMARY & RESULTS - ########################

# Plots a .png of the galaxy model summary, and sends it to the project dir.
def get_model_plot(galaxy_model):
    plot_model(galaxy_model,
               to_file='model_summary_plot.png',
               show_shapes=True,
               show_layer_names=True)


# Plots the accuracy and validation accuracy of the model
# so its more easily visualised
def plot_accuracy(history):
    accuracy_plot = plt
    accuracy_plot.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    accuracy_plot.title('model accuracy history')
    accuracy_plot.ylabel('Accuracy value (%)')
    accuracy_plot.xlabel('No. epoch')
    accuracy_plot.legend(['training set', 'validation set'], loc='upper left')
    accuracy_plot.show()
    return accuracy_plot


# Plots the loss and validation loss of the model.
def plot_loss(history):
    loss_plot = plt
    loss_plot.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    loss_plot.title('model loss history')
    loss_plot.ylabel('Loss value')
    loss_plot.xlabel('No. epoch')
    loss_plot.legend(['training set', 'validation set'], loc='upper left')
    loss_plot.show()
    return loss_plot

################################################################################

################################################################################

##################### - CHECKING PREDICTIONS - #################################

# def check_prediction(galaxy_model):
#     check_image = validation_images[0:1]
#     check_label = validation_labels[0:1]
#     predict = galaxy_model.predict(np.array(check_image))
#     output = {0: 'anti-clockwise spiral', 1: 'clockwise spiral',
#               2: 'edge on', 3: 'elliptical', 4: 'merging'}
#
#     print("Actual :- ", check_label)
#     print("Predicted :- ", output[np.argmax(predict)])

################################################################################

########################## - RUNNING MODELS - ##################################

########################## - RUNNING MODELS - ##################################

def base_model():
    b_model = create_galaxy_model_base()
    # b_model = tf.keras.models.load_model(
    # "C:/Users/PietroPC/OneDrive - University of Essex/University/Year 3/
    # CE301 Capstone Final Project/git/ce301_carneiro_da_silva_pietro_e/
    # galaxy_project/preloaded_checkpoints_base/checkpoint-0001.hdf5")
    b_model.summary()
    hist = get_model_fit(b_model, epochs=50, initial_epoch=0)
    get_model_eval(b_model)
    get_model_plot(b_model)
    plot_accuracy(hist)
    plot_loss(hist)


def vgg16_model():
    v_model = create_galaxy_model_vgg16()
    # v_model = tf.keras.models.load_model(
    # "C:/Users/PietroPC/OneDrive - University of Essex/University/Year 3/
    # CE301 Capstone Final Project/git/ce301_carneiro_da_silva_pietro_e/
    # galaxy_project/preloaded_checkpoints_vgg16/checkpoint-0001.hdf5")
    v_model.summary()
    hist = get_model_fit(v_model, epochs=10, initial_epoch=0)
    get_model_eval(v_model)
    get_model_plot(v_model)
    plot_accuracy(hist)
    plot_loss(hist)


def modified_vgg16_model():
    modified_b_model = create_galaxy_model_modified_vgg16()
    # modified_b_model = tf.keras.models.load_model(
    # "C:/Users/PietroPC/OneDrive - University of Essex/University/Year 3/
    # CE301 Capstone Final Project/git/ce301_carneiro_da_silva_pietro_e/
    # galaxy_project/preloaded_checkpoints_modified_base/checkpoint-0001.hdf5")
    modified_b_model.summary()
    hist = get_model_fit(modified_b_model, epochs=10, initial_epoch=0)
    get_model_eval(modified_b_model)
    get_model_plot(modified_b_model)
    plot_accuracy(hist)
    plot_loss(hist)


def modified_sequential_model():
    s_model = create_galaxy_model_modified_sequential()
    # modified_s_model = tf.keras.models.load_model(
    # "C:/Users/PietroPC/OneDrive - University of Essex/University/Year 3/
    # CE301 Capstone Final Project/git/ce301_carneiro_da_silva_pietro_e/
    # galaxy_project/preloaded_checkpoints_scratch/checkpoint-0001.hdf5")
    s_model.summary()
    hist = get_model_fit(s_model, epochs=10, initial_epoch=0)
    get_model_plot(s_model)
    plot_accuracy(hist)
    plot_loss(hist)


# This section of code is just to show what the images in the dataset look like
# after data augmentation and preprocessing.
data_aug = Sequential()
data_aug.add(data_augmentation())
data_aug.add(data_preprocessing())

plt.figure(figsize=(10, 10))

for images, _ in training_dataset.take(1):
    for i in range(9):
        augmented_images = data_aug(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

plt.show()

# base_model()

vgg16_model()

# modified_vgg16_model()

# modified_sequential_model()

################################################################################
