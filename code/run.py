import hyperparameters as hp
from datetime import datetime
import tensorflow as tf
from models import VGGModel
from preprocess import split_train_validation_data, load_test_data

def train(model, train_data, train_labels, val_data, val_labels, init_epoch):
    print("inside train")

    # shuffle training data 
    rand_i = tf.random.shuffle(range(len(train_labels)))
    train_data = tf.gather(train_data, rand_i)
    train_labels = tf.gather(train_labels, rand_i)

    model.fit(
        x=train_data, 
        y=train_labels,
        validation_data=(val_data, val_labels),
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=None,
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
    
    train(model, train_data, train_labels, val_data, val_labels, init_epoch)
    print("done training")

    test(model, test_data, test_labels)
    print("done testing")

main()