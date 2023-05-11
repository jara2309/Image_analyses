import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # #data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)#path naar directory
    # #data_dir = pathlib.Path(data_dir)
    # # image_count = len(list(data_dir.glob('*/*/*.jpg')))
    # # print(image_count)
    # # roses = list(data_dir.glob('train/tumor/*'))
    # # hi = PIL.Image.open(str(roses[0]))
    # # # hi.show()
    # # roos2 = PIL.Image.open(str(roses[1]))
    # # # roos2.show()
    # # tulips = list(data_dir.glob('test/no_tumor/*'))
    # # tullip1 = PIL.Image.open(str(tulips[0]))
    # # # tullip1.show()
    # # tullip2 = PIL.Image.open(str(tulips[1]))
    # # # tullip2.show()
    #
    batch_size = 16
    img_height = 180
    img_width = 180
    # train_data_dir = "C:/Users/jaral/PycharmProjects/Image_analyses_ours/train"
    # traindata_dir = pathlib.Path(train_data_dir)
    #
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     traindata_dir,
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)
    #
    # validation_data_dir = "C:/Users/jaral/PycharmProjects/Image_analyses_ours/val"
    # validatedata_dir = pathlib.Path(validation_data_dir)
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     validatedata_dir,
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)
    #
    # class_names = train_ds.class_names
    #
    # print(class_names)
    #
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #     plt.show()
    #
    # for image_batch, labels_batch in train_ds:
    #     print(image_batch.shape)
    #     print(labels_batch.shape)
    #     break
    #
    # AUTOTUNE = tf.data.AUTOTUNE
    #
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #
    # normalization_layer = layers.Rescaling(1. / 255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # print(np.min(first_image), np.max(first_image))
    #
    # num_classes = len(class_names)
    #
    # model = Sequential([
    #     layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes)
    # ])
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # model.summary()
    #
    # epochs = 10
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs_range = range(epochs)
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()
    #
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.RandomFlip("horizontal",
    #                           input_shape=(img_height,
    #                                        img_width,
    #                                        3)),
    #         layers.RandomRotation(0.1),
    #         layers.RandomZoom(0.1),
    #     ]
    # )
    #
    # plt.figure(figsize=(10, 10))
    # for images, _ in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.axis("off")
    #     plt.show()
    #
    # model = Sequential([
    #     data_augmentation,
    #     layers.Rescaling(1. / 255),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes, name="outputs")
    # ])
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # model.summary()
    # epochs = 15
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs_range = range(epochs)
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()
    #
    # convert = tf.lite.TFLiteConverter.from_keras_model(model)
    # convert_model = convert.convert()
    # with open('keras_model.tflite', 'wb') as f:
    #     f.write(convert_model)
    #

    def roc_curvee():
        """
        Calculates the roc curve based on the saved keras model.
        """
        # Path to test directory
        test_path = "C:/Users/jaral/PycharmProjects/Image_analyses_ours/test"
        directory_test_list = os.listdir(test_path)
        # Declare lists
        test_dataset_y_values = []
        predictions = []
        # Declare variables
        img_height = 180
        img_width = 180

        # Get saved model
        model_interpreter = tf.lite.Interpreter(model_path="keras_model.tflite")
        # Optimizes inference
        model_interpreter.allocate_tensors()

        # Get info out the model
        input_details_model = model_interpreter.get_input_details()
        output_details_model = model_interpreter.get_output_details()

        # Loops through test directory to give 1 point to tumor pictures
        for sub_directory in directory_test_list:
            if 'NORMAL' in str(sub_directory):
                tumor = 0
            elif 'COVID19' in str(sub_directory):
                tumor = 1
            else:
                tumor = 2
            # Updates the directory path
            sub_directory_full_path = test_path + '/' + sub_directory
            # Loops through all the images in the test directory and gives them a score
            for file_image in os.listdir(sub_directory_full_path):
                image_path = sub_directory_full_path + '/' + file_image
                # Prepares images for testing model
                img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                # Put the trained model against images
                model_interpreter.set_tensor(input_details_model[0]['index'], img_array)
                model_interpreter.invoke()
                output = model_interpreter.get_tensor(output_details_model[0]['index'])
                # Calculates score
                score = tf.nn.softmax(output[0])
                test_dataset_y_values.append(tumor)
                predictions.append(score[1])
        try:
            roc_auc_score(test_dataset_y_values, predictions)
        except ValueError:
            pass
        # Calculates auc_score
        auc_score = roc_auc_score(test_dataset_y_values,predictions, multi_class='ovr')

        # Calculates roc_curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_dataset_y_values, predictions)
        # Plot roc_curve
        plt.plot(false_positive_rate, true_positive_rate, label='ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC plot Tosca & Jara')
        plt.legend()

        # Add AUC score to the plot
        plt.text(0.7, 0.2, f'AUC = {auc_score:.2f}')

        plt.show()


    roc_curvee()
