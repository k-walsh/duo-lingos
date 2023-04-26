import hyperparameters as hp
from datetime import datetime
import tensorflow as tf
from models import VGGModel
from preprocess import split_train_validation_data, load_test_data
from tensorboard_utils import CustomModelSaver
import os

def train(model, train_data, train_labels, val_data, val_labels, checkpoint_path, logs_path, init_epoch):
    print("inside train")

    # shuffle training data 
    rand_i = tf.random.shuffle(range(len(train_labels)))
    train_data = tf.gather(train_data, rand_i)
    train_labels = tf.gather(train_labels, rand_i)

    # shuffle val data
    rand_i_val = tf.random.shuffle(range(len(val_labels)))
    val_data = tf.gather(val_data, rand_i_val)
    val_labels = tf.gather(val_labels, rand_i_val)

    callback_list = [
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=logs_path,
        #     update_freq='batch',
        #     profile_batch=0),
        # ImageLabelingLogger(logs_path, datasets),
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

def test(model, test_data, test_labels):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        y=test_labels,
        verbose=1,
    )

def main():
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    checkpoint_path = "checkpoints" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    os.makedirs(checkpoint_path)

    print("loading data")

    train_data, train_labels, val_data, val_labels = split_train_validation_data()
    print(f"{len(train_data)} training samples loaded")
    print(f"{len(val_data)} validation samples loaded")

    test_data, test_labels = load_test_data()
    print(f"{len(test_data)} testing samples loaded")

    model = VGGModel()
    model(tf.keras.Input(shape=(200, 200, 3)))

    # Print summaries for both parts of the model
    model.vgg16.summary()
    model.head.summary()

    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    train(model, train_data, train_labels, val_data, val_labels, checkpoint_path, logs_path, init_epoch)
    print("done training")

    test(model, test_data, test_labels)
    print("done testing")

main()