from sklearn.preprocessing import LabelBinarizer
import os 
import random
import numpy as np
from skimage import io, img_as_float32
import matplotlib.pyplot as plt



# train_df = pd.read_csv("sign_mnist_train.csv")
# test_df = pd.read_csv("sign_mnist_test.csv")

# y_train = train_df['label']
# y_test = test_df['label']
# del train_df['label']
# del test_df['label']

# need to load in data from images

# file_list = []
# for root, _, files in os.walk('/asl_data/asl_alphabet_train'):
#     for name in files:
#         if name.endswith(".jpg"):
#             file_list.append(os.path.join(root, name))

# https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']

def load_train_data():
    """load the training data and returns 2 numpy arrays of the training data and labels"""
    train_data = []
    train_labels = []
    for category in categories:
        category_path = os.path.join('data/train', category)
        print(category)
        class_letter = categories.index(category)
        for img in os.listdir(category_path):
            file_path = os.path.join(category_path,img)
            img_array = img_as_float32(io.imread(file_path))
            assert img_array.shape == (200,200,3)
            train_data.append(img_array)
            train_labels.append(class_letter)
    return np.array(train_data), np.array(train_labels)

# uncomment when ready to load
# train_data, train_labels = load_train_data()
# assert len(train_data) == len(train_labels)
# print(f"{len(train_data)} training samples loaded")
# plt.imshow(train_data[100])


# TODO: need to randomly shuffle the training samples - tf.gather??? - do we want tf variables or np arrays??

# TODO: augment training data and one hot encode label vectors??
# standardize too? like in hw 5?

# TODO: do we care about images being rbg vs black white ??

# TODO: make validation set from training set (after shuffle pick 20%)

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


# TODO: maybe save these arrays as csvs so we can just read them in and don't have to do all this preprocessing each time


# label_binarizer = LabelBinarizer()
# y_train = label_binarizer.fit_transform(y_train)
# y_test = label_binarizer.fit_transform(y_test)

# x_train = train_df.values
# x_test = test_df.values

# x_train = x_train / 255
# x_test = x_test / 255

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)