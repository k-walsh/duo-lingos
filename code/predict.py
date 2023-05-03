import os

import tensorflow as tf
from tensorflow import keras
from models import VGGModel
from preprocess import Datasets
from skimage import io, img_as_float32

def process_image(image_path):
    img = img_as_float32(io.imread(image_path))
    print(img.shape)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    print(img.shape)
    return img

def predict_handshape(img_path, model_checkpoint_path):
    model = VGGModel()
    model(tf.keras.Input(shape=(200, 200, 3)))

    model.vgg16.summary()
    model.head.summary()

    model.vgg16.load_weights('code/vgg16_imagenet.h5', by_name=True)
    model.head.load_weights(model_checkpoint_path, by_name=False)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    img = process_image(img_path)
    print(img.shape)
    pred = model.predict(img, batch_size=2, verbose=1)
    # TODO: erroring because shape is wrong but idk why it should be (batch_sz,200,200,3) ??

    print(pred)
    return pred

def main():
    predict_handshape('data/test/B_test.jpg', 'checkpoints/vgg_model/050223-210825/vgg.weights.e003-acc0.9770.h5')

main()