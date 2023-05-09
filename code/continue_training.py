
import tensorflow as tf
from tensorflow import keras
from models import VGGModel
from preprocess import Datasets
from skimage import io, img_as_float32
import numpy as np
import datetime
import os
from run import train

def continue_training(model_checkpoint_path):
    """Load and compile the model from the checkpoint and continue training"""

    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    datasets = Datasets()

    checkpoint_path = "checkpoints" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    os.makedirs(checkpoint_path)

    model = VGGModel()
    model(tf.keras.Input(shape=(200, 200, 3)))

    # model.vgg16.summary()
    # model.head.summary()

    model.vgg16.load_weights('code/vgg16_imagenet.h5', by_name=True)
    model.head.load_weights(model_checkpoint_path, by_name=False)

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    
    train(model, datasets, checkpoint_path, logs_path, init_epoch)
    
    return model

print('continuing to train')
continue_training('checkpoints/vgg_model/050423-130714/vgg.weights.e004-acc0.4485.h5')
print('done training')