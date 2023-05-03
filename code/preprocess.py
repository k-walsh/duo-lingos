from sklearn.preprocessing import LabelBinarizer
import os 
import random
import numpy as np
from skimage import io, img_as_float32
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import hyperparameters as hp
from tensorflow.python.keras import utils as ut
from PIL import Image

class Datasets():
    def __init__(self):
        
        # https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/
        self.categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
        self.label_binarizer = LabelBinarizer()

        self.classes = [""] * hp.num_classes

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        self.train_data, self.val_data = self.load_data()
        # self.train_data, self.train_labels, self.val_data, self.val_labels = self.split_train_validation_data()

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img

    def load_data(self):
        "inside load data"
        
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn,
            validation_split=0.2)
        img_size = hp.img_size
        classes_for_flow = None

        print("data gen made")

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Directory containing the images
        self.validate_images('data/train')

        # Form image data generator from directory structure
        # just going to load in the training data and then split - the 28 test images aren't worth the hassle
        train_data_gen = data_gen.flow_from_directory(
            'data/train',
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=True,
            classes=classes_for_flow,
            subset='training')
        
        print("train data gen made")



        val_data_gen = data_gen.flow_from_directory(
            'data/train',
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=hp.batch_size,
            shuffle=True,
            classes=classes_for_flow,
            subset='validation')
        
        print("val training gen made")
        
        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir('data/train'):
                if os.path.isdir(os.path.join('data/train', dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[train_data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(train_data_gen.class_indices[img_class])
                self.classes[int(train_data_gen.class_indices[img_class])] = img_class

        # print("ids to classes", self.idx_to_class)
        # print("class to id", self.class_to_idx)
        # print("classes", self.classes)
        # print("train gen", train_data_gen)
        # print("train paths", train_data_gen.filepaths[3000])
        # print("train labels", train_data_gen.labels[3000])
        # print("val gen", val_data_gen)

        return train_data_gen, val_data_gen
    
    def validate_images(self, directory):
        # Directory containing the images

        # Supported image extensions
        extensions = ('.jpg')

        # Iterate through the files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(extensions):
                # Load the image and get its format
                filepath = os.path.join(directory, filename)
                try:
                    with Image.open(filepath) as img:
                        format = img.format
                        print(f'{filename}: {format}')
                except OSError:
                    print(f'{filename} cannot be opened')
                    # Delete the file if it cannot be opened
                    os.remove(filepath)
            else:
                print(f'{filename} has an unsupported extension')
                # Delete the file if it has an unsupported extension
                os.remove(os.path.join(directory, filename))

        print ("images validated")

    # def load_test_data(self):
    #     """load the testing data and return numpy arrays of data and encoded labels"""
    #     # Get list of all images in testing directory
    #     test_file_list = []
    #     for root, _, files in os.walk(os.path.join("data/test/")):
    #         for name in files:
    #             test_file_list.append(os.path.join(root, name))

    #     # Shuffle filepaths
    #     random.shuffle(test_file_list)

    #     test_data = []
    #     test_labels = []
    #     for file_path in test_file_list:
    #         letter = file_path.split("/")[2].split("_")[0]
    #         img_array = img_as_float32(io.imread(file_path))
    #         assert img_array.shape == (200,200,3)
    #         test_data.append(img_array)
    #         test_labels.append(letter)

    #     test_data, test_labels = np.array(test_data), np.array(test_labels)
    #     test_labels = self.label_binarizer.fit_transform(test_labels) # to one hot encode the labels

    #     return test_data, test_labels

