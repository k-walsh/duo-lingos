from sklearn.preprocessing import LabelBinarizer
import os 
import cv2


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

training_data = []
for category in categories:
    path = os.path.join('data/train', category)
    print(category)
    class_num = categories.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path,img))
      new_array = cv2.resize(img_array, (200, 200))
      training_data.append([new_array, class_num])

print(f"done loading training data, {len(training_data)} training examples loaded")



# label_binarizer = LabelBinarizer()
# y_train = label_binarizer.fit_transform(y_train)
# y_test = label_binarizer.fit_transform(y_test)

# x_train = train_df.values
# x_test = test_df.values

# x_train = x_train / 255
# x_test = x_test / 255

# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)