import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


def initializing_data():
    """ In this function all the data has been provided to use for
    further analyzing.

    :return: train_dataset - list - List with all the images of the
    training dataset
    :return: val_dataset - list - List with all the images of the
    validation dataset
    :return: classes_names - list - List with the names of the classes
    :return: image_height - int - Int to set the heights of the images
    :return: image_width - int - Int to set the width of the images
    """
    # Setting the batch size
    batch_size = 128
    # Setting the image size
    image_height = 180
    image_width = 180
    # Defining the location of the trainingset to use
    train_data_dir = "train"
    traindata_dir = pathlib.Path(train_data_dir)
    # Gets the images from the given location to form the training set
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        traindata_dir,
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size)

    # Defining the location of the validationset to use
    validation_data_dir = "val"
    validatedata_dir = pathlib.Path(validation_data_dir)
    # Gets the images from the given location to form the validation set
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        validatedata_dir,
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size)

    # Creating a variable to set the names of the different classes
    classes_names = train_dataset.class_names

    # Returning the variables to use in other functions
    return train_dataset, val_dataset, \
           classes_names, image_height, image_width


def pre_processing(training_dataset, validation_dataset, classes_names,
                   image_height, image_width):
    """In this function the training and validationset will be
    pre-processed by using a normalisation layer and augmentation on
    the data.

    :param training_dataset - list - List with all the images of the
    training dataset
    :param validation_dataset - list - List with all the images of the
    validation dataset
    :param classes_names - str - String with the names of the classes
    :param image_height - int - Int to set the heights of the images
    :param image_width - int - Int to set the width of the images

    :return train_data - list - List with all the images of the training
    dataset after pre-processing
    :return val_data - list - List with all the images of the validation
    dataset after pre-processing
    :return classes_numbers - list - List with the names of the classes
    :return augmentation_data - -
    """
    # Setting the decision about the level of parallelism to use
    autotune = tf.data.AUTOTUNE
    # Using the autotune on the training and validation datasets
    train_data = training_dataset.cache().shuffle(1000).prefetch(
        buffer_size=autotune)
    val_data = validation_dataset.cache().prefetch(buffer_size=autotune)

    # Initalizing the normalisation layer
    normalization_layer = layers.Rescaling(1. / 255)
    # Using the normalisation layer on the training dataset
    normalized_ds = train_data.map(
        lambda x, y: (normalization_layer(x), y))
    # Adds normalisation for every picture
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    # Turning the class names into class numbers by using the length
    classes_numbers = len(classes_names)

    # Using augmentation by using 3 different layers
    augmentation_data = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(image_height,
                                           image_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Showing a plot with the first 9 images of the training dataset
    plt.figure(figsize=(10, 10))
    for images, _ in train_data.take(1):
        for i in range(9):
            plt.axis("off")
        # Showing the image
        plt.show()

    return train_data, val_data, classes_numbers, augmentation_data


def model_training(train_dataset, val_dataset, number_classes,
                   augmentation_data):
    """ In this function the actual model will be trained by using the
    training and validation dataset.

    :param train_dataset - list - List with all the images of the
    training dataset
    :param val_dataset - list - List with all the images of the
    validation dataset
    :param number_classes - list - List with all the image classes by
    using numbers for every class
    :param augmentation_data -  -

    """
    # Create a learning rate scheduler callback
    model = Sequential([
        augmentation_data,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_classes, name="outputs")

    ])

    # Viewing the training and validation accuracy for each epoch
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # Viewing all the layers of the network of the model
    model.summary()

    # Setting the number of epochs to use
    epochs = 15
    # Training the model for the given number of epochs with the
    # training and validation dataset
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
    )
    # Creating plots with the training and validation loss and accuracy
    acc = history.history['accuracy'][2:]
    val_acc = history.history['val_accuracy'][2:]
    loss = history.history['loss'][2:]
    val_loss = history.history['val_loss'][2:]
    # Setting the range for the x-axis
    epochs_range = range(2, epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    convert = tf.lite.TFLiteConverter.from_keras_model(model)
    convert_model = convert.convert()
    with open('keras_model9.tflite', 'wb') as f:
        f.write(convert_model)


def auc_score():
    """ In this function the AUC-score of the model will be calculated
    by using the test dataset. The AUC-score will be calculated after
    testing the test dataset on the created model.
    """
    # Defining the location of the testset to use
    test_path = "test"
    directory_test_list = os.listdir(test_path)
    # Declaring the lists
    test_dataset_y_values = []
    predictions = []
    # Declaring the variables
    image_height = 180
    image_width = 180
    y = 0

    # Get saved model
    model_interpreter = tf.lite.Interpreter(
        model_path="keras_model9.tflite")
    # Optimizes inference
    model_interpreter.allocate_tensors()

    # Get info out the model
    input_details_model = model_interpreter.get_input_details()
    output_details_model = model_interpreter.get_output_details()

    # Loops through test directory to give 0 points for buffalo,
    # 1 point for elephant, 2 point for rhino or 3 points for zebra
    for sub_directory in directory_test_list:
        if 'Buffalo' in str(sub_directory):
            animal = 0
        elif 'Elephant' in str(sub_directory):
            animal = 1
        elif 'Rhino' in str(sub_directory):
            animal = 2
        elif 'Zebra' in str(sub_directory):
            animal = 3
        else:
            animal = -1
        # Updates the directory path
        sub_directory_full_path = test_path + '/' + sub_directory
        # Loops through all the images in the test directory
        # and gives them a score
        for file_image in os.listdir(sub_directory_full_path):
            image_path = sub_directory_full_path + '/' + file_image
            # Prepares images for testing model
            img = tf.keras.utils.load_img(image_path, target_size=(
                image_height, image_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            # Put the trained model against images
            model_interpreter.set_tensor(
                input_details_model[0]['index'], img_array)
            model_interpreter.invoke()
            output = model_interpreter.get_tensor(
                output_details_model[0]['index'])
            # Calculates score
            scores = tf.nn.softmax(output[0])
            test_dataset_y_values.append(animal)
            y = label_binarize(test_dataset_y_values,
                               classes=[0, 1, 2, 3, 4])

            predictions.append(scores)
    try:
        roc_auc_score(y, predictions)
    except ValueError:
        pass
    # Calculates auc_score
    auc_score_model = roc_auc_score(y, predictions, multi_class='ovr')
    print(auc_score_model)


if __name__ == '__main__':
    training_data, validation_data, \
    class_names, img_height, img_width = initializing_data()
    train_ds, val_ds, num_classes, data_augmentation = pre_processing(
        training_data, validation_data, class_names, img_height,
        img_width)
    model_training(train_ds, val_ds, num_classes, data_augmentation)
    auc_score()
