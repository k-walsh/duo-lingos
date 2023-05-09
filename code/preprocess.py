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

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return img

    def load_data(self):
        "inside load data"
        
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=5,
            brightness_range=[0.8,1.2],
            width_shift_range=0.2, 
            height_shift_range=0.2,
            zoom_range=0.2,
            preprocessing_function=self.preprocess_fn,
            validation_split=0.2)
        img_size = hp.img_size
        classes_for_flow = None

        print("data gen made")

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
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

        return train_data_gen, val_data_gen
