# ASL Alphabet Translator 
# by the DuoLingos (Kiera Walsh and Joan Nekoye)

This project's goal was to create a live ASL alphabet translator that takes live user input and classifies the handshape as a letter. The classifier is a deep learning model with vgg16 as its base and a custom head with 29 output classes (for the 26 letters and signs for nothing, space, and delete). Dropout layers and data augmentation were added to reduce overfitting. We were able to reach approximately 61% testing accuracy with this model.

The dataset used was from Kaggle and consisted of many images of each handeshape. It can be viewed [here]([url](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)). Due to the large size of the dataset and limited computing resources for training, the dataset was cut down to 200 images per letter with a total of 5,800 images. These were divided into a train and test set.

To run the program, run `python3 code/camera.py` in the terminal and this should open a new window with a live video feed. Place your hand in the blue box and try out different signs. The yellow letter on the screen will display the corresponding letter for that handshape. As is standard practice in ASL, this program works best with a clear background that is free from distractions. Note that the program is not 100% accurate, refer to the picture below for correct handshapes:

![chart of asl alphabet handshape](https://ecdn.teacherspayteachers.com/thumbitem/American-Sign-Language-Alphabet-Chart-1604401288/original-142784-1.jpg)

Technical program requirements include those in the cs1430 course environment, such as tensorflow and opencv2.
