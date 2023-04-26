from sklearn.preprocessing import LabelBinarizer
import os 
import random
import numpy as np
from skimage import io, img_as_float32
import matplotlib.pyplot as plt
import tensorflow as tf

class Datasets():
    def __init__(self):
        
        # https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/
        self.categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
        self.label_binarizer = LabelBinarizer()

        self.train_data, self.train_labels, self.val_data, self.val_labels = self.split_train_validation_data()
        self.test_data, self.test_labels = self.load_test_data()

    def load_train_data(self):
        """load the training data and returns 2 numpy arrays of the training data and labels"""
        train_data = []
        train_labels = []
        start_start = time.time()
        for category in categories:
            start = time.time()
            category_path = os.path.join('data/train', category)
            i = 0
            for img in os.listdir(category_path):
                # if i > 20: # to only load some train images for now
                #     break
                file_path = os.path.join(category_path,img)
                img_array = img_as_float32(io.imread(file_path))
                assert img_array.shape == (200,200,3)
                train_data.append(img_array)
                train_labels.append(category)
                i += 1
            end = time.time()
            print(category, end-start, "seconds")
        end_end = time.time()
        print(f"total time to load data {end_end - start_start} seconds")
        return np.array(train_data), np.array(train_labels)

    def split_train_validation_data():
    """splits train into train and validation (still np arrays)"""
        train_data, train_labels = load_train_data()
        print("inside split, loaded data")
        train_labels = label_binarizer.fit_transform(train_labels)
        print("one hot encoded the labels")

        dataset_size = len(train_labels)
        train_size = int(dataset_size * 0.80)
        val_size = dataset_size - train_size  

        # randomly choose 20% indices to be indices of the validation data
        rand_indices = random.sample(range(1, dataset_size), val_size)
        val_data = train_data[rand_indices]
        val_labels = train_labels[rand_indices]
        print("split train and val data")

        train_data = np.delete(train_data, rand_indices, axis=0)
        train_labels = np.delete(train_labels, rand_indices, axis=0)

        assert len(train_data) + len(val_data) == dataset_size
        assert len(train_labels) + len(val_labels) == dataset_size

        return train_data, train_labels, val_data, val_labels

    # TODO: potentially augment training data and one hot encode label vectors?? standardize too? like in hw 5?
    # TODO: do we care about images being rbg vs black white ??
    # TODO: divide images by 255??? normalize

    def load_test_data(self):
        """load the testing data and return numpy arrays of data and encoded labels"""
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

        test_data, test_labels = np.array(test_data), np.array(test_labels)
        test_labels = self.label_binarizer.fit_transform(test_labels) # to one hot encode the labels

        return test_data, test_labels

# TODO: maybe save these arrays as csvs so we can just read them in and don't have to do all this preprocessing each time
