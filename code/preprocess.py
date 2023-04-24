from sklearn.preprocessing import LabelBinarizer
import os 
import random
import numpy as np
from skimage import io, img_as_float32
import matplotlib.pyplot as plt
import tensorflow as tf

# https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
label_binarizer = LabelBinarizer()

def load_train_data():
    """load the training data and returns 2 numpy arrays of the training data and labels"""
    train_data = []
    train_labels = []
    for category in categories:
        category_path = os.path.join('data/train', category)
        print(category)
        # class_letter = categories.index(category)
        i = 0
        for img in os.listdir(category_path):
            if i > 100: # to only load 100 train images for now
                break
            file_path = os.path.join(category_path,img)
            img_array = img_as_float32(io.imread(file_path))
            assert img_array.shape == (200,200,3)
            train_data.append(img_array)
            train_labels.append(category)
            i += 1
    return np.array(train_data), np.array(train_labels)

def load_train_validation_data():
    """
    * shuffles the output of load_train_data so everything is randomly shuffled
    * splits the training data into validation and train sets
    * returns the train data, train labels, validation data, and validation labels as tensors
    """
    train_data, train_labels = load_train_data()
    train_labels = label_binarizer.fit_transform(train_labels)

    # shuffle the data so not in alphabet order - tf.gather shuffles the data & labels together
    rand_i = tf.random.shuffle(range(len(train_labels)), seed=12)
    all_train_data = tf.gather(train_data, rand_i)
    all_train_labels = tf.gather(train_labels, rand_i)

    dataset_size = len(all_train_labels)
    train_size = int(dataset_size * 0.80)
    val_size = int(dataset_size * 0.20)
    assert train_size + val_size == dataset_size

    train_data = all_train_data.take(train_size)  
    train_labels = all_train_labels.take(train_size)

    val_data = all_train_data.skip(train_size).take(val_size)
    val_labels = all_train_labels.skip(train_size).take(val_size)

    return train_data, train_labels, val_data, val_labels

# TODO: potentially augment training data and one hot encode label vectors?? standardize too? like in hw 5?

# TODO: do we care about images being rbg vs black white ??


def load_test_data():
    """load the testing data and return numpy arrays of data and labels"""
    # Get list of all images in testing directory
    test_file_list = []
    for root, _, files in os.walk(os.path.join("data/test/")):
        for name in files:
            test_file_list.append(os.path.join(root, name))

    # Shuffle filepaths
    random.shuffle(test_file_list)

    test_data = []
    test_labels = []
    for file_path in test_file_list:
        letter = file_path.split("/")[2].split("_")[0]
        img_array = img_as_float32(io.imread(file_path))
        assert img_array.shape == (200,200,3)
        test_data.append(img_array)
        test_labels.append(letter)

    return np.array(test_data), np.array(test_labels)

test_data, test_labels = load_test_data()
assert len(test_data) == len(test_labels)
print(f"{len(test_data)} testing samples loaded")

plt.imshow(test_data[17])
plt.show()
print(test_labels[17])

test_labels = label_binarizer.fit_transform(test_labels)


# TODO: maybe save these arrays as csvs so we can just read them in and don't have to do all this preprocessing each time

train_data, train_labels, val_data, val_labels = load_train_validation_data()