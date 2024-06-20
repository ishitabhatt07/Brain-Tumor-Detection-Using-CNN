import glob

import tensorflow as tf
import keras
from PIL import Image
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
from os import listdir


def crop_brain_contour(image, plot=False):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # =series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Finding contours in thresholded image, then grabbing the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Finding the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # cropping new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')

        plt.show()

    return new_image

ex_img = cv2.imread('C:\\Users\\pakhi\\PycharmProjects\\MLproject\\Brain-Tumor-Detection-master\\augmented data\\no\\aug_10 no_0_383.jpg')
ex_new_img = crop_brain_contour(ex_img, True)

#LOADING THE DATASET-------------------------------------
def load_data(dir_list, image_size):
    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size

    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            image = cv2.imread(directory + '\\' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y

augmented_path = 'C:\\Users\\pakhi\\OneDrive\\Desktop\\college shit\\Brain-Tumor-Detection-master\\augmented data'

# augmented data (yes and no) contains both the original and the new generated examples
augmented_yes = augmented_path + '\\yes'
augmented_no = augmented_path + '\\no'

IMG_WIDTH, IMG_HEIGHT = (240, 240)

X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))


# PLOTTING THE IMAGES--------------
def plot_sample_images(X, y, n=50):
    """
    Plots n sample images for both values of y (labels).
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    for label in [0, 1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n / columns_n)

        plt.figure(figsize=(20, 10))

        i = 1  # current plot
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # remove ticks
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()

plot_sample_images(X, y)


#SPLITTING THE DATA-------------------------

def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of development examples = " + str(X_val.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_val (dev) shape: " + str(X_val.shape))
print ("Y_val (dev) shape: " + str(y_val.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))


# Nicely formatted time string------------
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"


def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)

    score = f1_score(y_true, y_pred)

    return score


#BUILDING THE MODEL---------------------

def build_model(input_shape):
    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)  # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)  # shape=(?, 238, 238, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X)  # shape=(?, 59, 59, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X)  # shape=(?, 14, 14, 32)

    # FLATTEN X
    X = Flatten()(X)  # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X)  # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.

    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

    return model

IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)
model = build_model(IMG_SHAPE)
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard
log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')
# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath="cnn-parameters-improvement-{epoch:02d}-{val_acc:.2f}"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(
    filepath="models/cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.model",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

'''

#TRAIN THE MODEL--------------

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")

#TRAINING FOR MORE EPOCHS

start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")
start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")
start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")
'''
start_time = time.time()

model.fit(x=X_train, y=y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

end_time = time.time()
execution_time = (end_time - start_time)
print(f"Elapsed time: {hms_string(execution_time)}")
history = model.history.history
for key in history.keys():
    print(key)


def plot_metrics(history):

    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.ylim([0, 1.5])
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.ylim([0, 1.5])
    plt.legend()
    plt.show()

    print("Train Loss:", train_loss)
    print("Validation Loss:", val_loss)
    print("Train Accuracy:", train_acc)
    print("Validation Accuracy:", val_acc)


plot_metrics(history)



best_model = load_model(filepath='models/cnn-parameters-improvement-03-0.89.model')
best_model.metrics_names

'''
#-------------------------------------------------------------------------------------------------------------


  # Replace 'brain_tumor_detection_model.h5' with your model file

# Directory paths for images with and without tumors
no_folder = 'C:\\Users\\pakhi\\PycharmProjects\\MLproject\\Brain-Tumor-Detection-master\\augmented data\\no'  # Replace 'path/to/yes/folder' with the actual path to the folder containing tumor images
yes_folder = 'C:\\Users\\pakhi\\PycharmProjects\\MLproject\\Brain-Tumor-Detection-master\\augmented data\\yes'    # Replace 'path/to/no/folder' with the actual path to the folder containing non-tumor images

def load_and_resize_images(folder_path, target_size=(240, 240)):
    resized_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming the images are in jpg or png format
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).resize(target_size)
            resized_images.append(np.array(image))
    return np.array(resized_images)

# Load and resize images from the 'no' folder
resized_no_images = load_and_resize_images(no_folder)

# Load and resize images from the 'yes' folder
resized_yes_images = load_and_resize_images(yes_folder)
# Function to load and preprocess images for predictions
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(240, 240))  # Assuming input shape of the CNN model is (224, 224, 3)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize pixel values to [0, 1]
    return img_array

# Generate predictions and true labels for images with tumors
tumor_image_paths = [os.path.join(yes_folder, f) for f in os.listdir(yes_folder)]
tumor_predictions = []
tumor_true_labels = []
for img_path in tumor_image_paths:
    preprocessed_img = load_and_preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    predicted_label = 1 if prediction > 0.5 else 0
    tumor_predictions.append(predicted_label)
    tumor_true_labels.append(1)

# Generate predictions and true labels for images without tumors
non_tumor_image_paths = [os.path.join(no_folder, f) for f in os.listdir(no_folder)]
non_tumor_predictions = []
non_tumor_true_labels = []
for img_path in non_tumor_image_paths:
    preprocessed_img = load_and_preprocess_image(img_path)
    prediction = model.predict(preprocessed_img)
    predicted_label = 1 if prediction > 0.5 else 0
    non_tumor_predictions.append(predicted_label)
    non_tumor_true_labels.append(0)

# Combine predictions and true labels for all images
predictions_list = tumor_predictions + non_tumor_predictions
true_labels_list = tumor_true_labels + non_tumor_true_labels

# Initialize lists to store individual metrics for each image
precision_list, recall_list, f1_list, specificity_list = [], [], [], []

# Loop through predictions and true labels to calculate metrics for each image
for true_labels_single, predictions_single in zip(true_labels_list, predictions_list):
    print("True Labels:", true_labels_single)
    print("Predictions:", predictions_single)
    precision = precision_score([true_labels_single], [predictions_single])
    recall = recall_score([true_labels_single], [predictions_single])
    f1 = f1_score([true_labels_single], [predictions_single])
    confusion_mat = confusion_matrix([true_labels_single], [predictions_single])

    # Extract values from confusion matrix
    if confusion_mat.shape == (2, 2):
        tn, fp, fn, tp = confusion_mat.ravel()
        specificity = tn / (tn + fp)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0

    # Append metrics for each image to respective lists
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    specificity_list.append(specificity)

# Create a DataFrame to store individual metrics for each image
metrics_df = pd.DataFrame({
    'Precision': precision_list,
    'Recall': recall_list,
    'F1 Score': f1_list,
    'Specificity': specificity_list
})

# Save the individual metrics to a CSV file
metrics_df.to_csv('individual_metrics_results.csv', index=False)

print("Individual metrics saved to 'individual_metrics_results.csv'.")
'''
#------------------------------------------------------------------------------------------------------------------------------------

loss, acc = best_model.evaluate(x=X_test, y=y_test)

print(f"Test Loss = {loss}")
print(f"Test Accuracy = {acc}")


import numpy as np

# Assuming X_test is a list of images, convert it to a NumPy array
X_test = np.array(X_test)

# Check the shape of your X_test array
print("Original X_test shape:", X_test.shape)


# Add channel dimension for single images (assuming color images)
X_test = np.expand_dims(X_test, axis=-1)

# Check the shape of your reshaped X_test array
print("Reshaped X_test shape:", X_test.shape)

batch_size = 32  # Set an appropriate batch size
y_pred_probs = model.predict(X_test, batch_size=batch_size)
y_pred_probs = model.predict(X_test)  # Predicted probabilities for binary classification
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)








# Calculate y_true (actual labels) based on the threshold of 0.5 for binary classification
y_true = y_test

# Now you have y_true and y_pred, you can use them to calculate various metrics
# For example, you can calculate accuracy, precision, recall, specificity, etc.

# Example code for accuracy calculation
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Example code for precision, recall, and specificity calculation (as shown in the previous response)
from sklearn.metrics import precision_score, recall_score, confusion_matrix

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])

print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)


# Extracting true positives, true negatives, false positives, and false negatives
true_negatives = conf_matrix[0, 0]
false_positives = conf_matrix[0, 1]
false_negatives = conf_matrix[1, 0]
true_positives = conf_matrix[1, 1]

# Print the results
print("True Negatives:", true_negatives)
print("False Positives:", false_positives)
print("False Negatives:", false_negatives)
print("True Positives:", true_positives)



y_test_prob = best_model.predict(X_test)
f1score = compute_f1_score(y_test, y_test_prob)
print(f"F1 score: {f1score}")


y_val_prob = best_model.predict(X_val)
f1score_val = compute_f1_score(y_val, y_val_prob)
print(f"F1 score: {f1score_val}")



def data_percentage(y):
    m = len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive

    pos_prec = (n_positive * 100.0) / m
    neg_prec = (n_negative * 100.0) / m

    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}")
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}")


# the whole data
data_percentage(y)
print("Training Data:")
data_percentage(y_train)
print("Validation Data:")
data_percentage(y_val)
print("Testing Data:")
data_percentage(y_test)

