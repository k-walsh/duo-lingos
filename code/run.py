import hyperparameters as hp
from datetime import datetime
import tensorflow as tf
from models import VGGModel
import os
from tensorboard_utils import \
        ImageLabelingLogger, CustomModelSaver
from preprocess import Datasets

def train(model, datasets, checkpoint_path, logs_path, init_epoch):

    # shuffle training data 
    rand_i = tf.random.shuffle(range(len(datasets.train_labels)))
    train_data = tf.gather(datasets.train_data, rand_i)
    train_labels = tf.gather(datasets.train_labels, rand_i)

    # shuffle val data
    rand_i_val = tf.random.shuffle(range(len(datasets.val_labels)))
    val_data = tf.gather(datasets.val_data, rand_i_val)
    val_labels = tf.gather(datasets.val_labels, rand_i_val)

    callback_list = [
        tf.keras.callbacks.TensorBoard(
             log_dir=logs_path,
             update_freq='batch',
             profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    model.fit(
        x=train_data, 
        y=train_labels,
        validation_data=(val_data, val_labels),
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )

def test(model, datasets):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=datasets.test_data,
        y=datasets.test_labels,
        verbose=1,
    )

def main():
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    datasets = Datasets()
    print("train data shape", datasets.train_data.shape)
    print("test data shape", datasets.test_data.shape)
    print("val data shape", datasets.val_data.shape)
    print("train labels shape", datasets.train_labels.shape)

    checkpoint_path = "checkpoints" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    os.makedirs(checkpoint_path)

    print(f"{len(datasets.train_data)} training samples loaded")
    print(f"{len(datasets.val_data)} validation samples loaded")

    print(f"{len(datasets.test_data)} testing samples loaded")

    model = VGGModel()
    model(tf.keras.Input(shape=(200, 200, 3)))

    # Print summaries for both parts of the model
    model.vgg16.summary()
    model.head.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    train(model, datasets, checkpoint_path, logs_path, init_epoch)
    print("done training")

    test(model, datasets)
    print("done testing")

main()