import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from keras.applications.vgg16 import VGG16
# from keras.models import Model

import hyperparameters as hp

class VGGModel(tf.keras.Model):
   def __init__(self):
      super(VGGModel, self).__init__()
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

      # self.vgg16 = VGG16(input_shape=(200,200,3),include_top=False,weights='imagenet')
      self.vgg16 = [
         # Block 1
         Conv2D(64, 3, 1, padding="same",
               activation="relu", name="block1_conv1"),
         Conv2D(64, 3, 1, padding="same",
               activation="relu", name="block1_conv2"),
         MaxPool2D(2, name="block1_pool"),
         # Block 2
         Conv2D(128, 3, 1, padding="same",
               activation="relu", name="block2_conv1"),
         Conv2D(128, 3, 1, padding="same",
               activation="relu", name="block2_conv2"),
         MaxPool2D(2, name="block2_pool"),
         # Block 3
         Conv2D(256, 3, 1, padding="same",
               activation="relu", name="block3_conv1"),
         Conv2D(256, 3, 1, padding="same",
               activation="relu", name="block3_conv2"),
         Conv2D(256, 3, 1, padding="same",
               activation="relu", name="block3_conv3"),
         MaxPool2D(2, name="block3_pool"),
         # Block 4
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block4_conv1"),
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block4_conv2"),
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block4_conv3"),
         MaxPool2D(2, name="block4_pool"),
         # Block 5
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block5_conv1"),
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block5_conv2"),
         Conv2D(512, 3, 1, padding="same",
               activation="relu", name="block5_conv3"),
         MaxPool2D(2, name="block5_pool")
      ]

      for layer in self.vgg16.layers:
            layer.trainable = False 

      self.head = [
         Flatten(),
         Dense(512, activation='relu'),
         Dropout(0.1),
         Dense(256, activation='relu'),
         Dense(128, activation='relu'),
         Dropout(0.1),
         Dense(hp.num_classes, activation='softmax')
      ]

      self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
      self.head = tf.keras.Sequential(self.head, name="vgg_head")
      
   def call(self, x):
      """ Passes the image through the network. """
      # print(x.shape) # <-- is (1, 200, 200, 3)
      x = self.vgg16(x)
      x = self.head(x)
      return x
    
   @staticmethod
   def loss_fn(labels, predictions):
      """ Loss function for model. """
      # sparse because labels are integers - from logits false bc softmax returns probabilities
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, predictions) 
      return loss
