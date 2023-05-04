import os

import tensorflow as tf
from tensorflow import keras
from models import VGGModel
from preprocess import Datasets
from skimage import io, img_as_float32
import numpy as np

data = Datasets()

def process_image(img):
    """takes in an img (as a np array) and processes it for the vgg model"""
    return tf.keras.applications.vgg16.preprocess_input(img)

def predict_handshape(img):
    model = VGGModel()
    model(tf.keras.Input(shape=(200, 200, 3)))

    # model.vgg16.summary()
    # model.head.summary()

    model.vgg16.load_weights('code/vgg16_imagenet.h5', by_name=True)
    model_checkpoint_path = 'checkpoints/vgg_model/050223-210825/vgg.weights.e003-acc0.9770.h5'
    model.head.load_weights(model_checkpoint_path, by_name=False)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    img = process_image(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, batch_size=4, verbose=1)

    max_i = np.argmax(pred)
    letter = data.idx_to_class[max_i]
    return letter