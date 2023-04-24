import tensorflow as tf
from keras.layers import \
       Conv2D, MaxPool2D, Flatten, Dense
from keras.applications.vgg16 import VGG16

import hyperparameters as hp

class VGGModel(tf.keras.Model):
    def __init__(self):
       super(VGGModel, self).__init__()

       self.optimizer =  tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
 
       self.vgg16 = VGG16(input_shape= (200,200,3),include_top=False,weights='imagenet')
    
       for layer in self.vgg16.layers:
              layer.trainable = False 
 
       self.head = [
            Flatten(),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(29, activation='softmax')
       ]
       self.head = tf.keras.Sequential(self.head, name="vgg_head")

    def call(self, x):
       """ Passes the image through the network. """

       x = self.vgg16(x)
       x = self.head(x)

       return x

    @staticmethod
    def loss_fn(labels, predictions):
       """ Loss function for model. """

       loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, predictions)

       return loss
