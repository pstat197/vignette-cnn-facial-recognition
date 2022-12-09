#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary Libraries

# In[ ]:


import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras import datasets, layers, models
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Split Data Into Training and Testing

# In[ ]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# ## Visualizing the Data Set

# In[69]:


# Checking the number of rows (records) and columns (features)
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(np.unique(train_labels))
print(np.unique(test_labels))

# Class Labels 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Visualizing some of the images from the training dataset

plt.figure(figsize=[10,10])
for i in range (25):    # for first 25 images
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[train_labels[i][0]])

plt.show()




# ## Changing the Data to Work with the Model

# In[108]:


# Standardizing/Normalizing is to convert all pixel values to values between 0 and 1.
# converting type to float is that to_categorical (one hot encoding) needs the data to be of type float by default.
# using to_categorical is that the loss function that we will be using in this code (categorical_crossentropy) when compiling the model needs data to be one hot encoded.

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
 
# Standardizing (255 is the total number of pixels an image can have)
train_images = train_images / 255
test_images = test_images / 255 


# ## Creating the CNN and Training the Model

# In[ ]:


# One hot encoding the target class (labels)
num_classes = 10
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)


#CNN Coding and Building the Layers

model = Sequential()

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))    # num_classes = 10

# Checking the model summary
model.summary()


# Optimizer used during Back Propagation for weight and bias adjustment - Adam (adjusts the learning rate adaptively).
# Loss Function used - Categorical Crossentropy (used when multiple categories/classes are present).
# Metrics used for evaluation - Accuracy.


model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=64, epochs=14,
                    validation_data=(test_images, test_labels))


# ## Visualizing the Accuracy of the Model

# In[ ]:


# Loss Curve - Comparing the Training Loss with the Testing Loss over increasing Epochs.
# Accuracy Curve - Comparing the Training Accuracy with the Testing Accuracy over increasing Epochs.
# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)


# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)


# ## Using the Model and Visualizing the Output

# In[ ]:


# Making the Predictions
pred = model.predict(test_images)
print(pred)

# Converting the predictions into label index 
pred_classes = np.argmax(pred, axis=1)


fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(test_images[i])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[np.argmax(test_labels[i])], class_names[pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)


# ## Functions created for next output visualization

# In[88]:


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"{class_names[int(predicted_label)]} {100*np.max(predictions_array):2.0f}% ({class_names[int(true_label)]})", 
               color=color)
    plt.show()

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, int(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.show()


# ### NEEDS TO BE IMPROVED: Showing the certainty of the predictions made by the model
# 
# Can't figure out why subplot isn't putting the images side by side

# In[111]:


num_rows = 8
num_cols = 5
num_images = num_rows * num_cols
i = 1
plt.subplot(2,2,1)
plot_image(i, pred[i], test_labels, test_images)
plt.subplot(2,2,2)
plot_value_array(i, pred[i], test_labels)
i = 2
plt.subplot(2,2,3)
plot_image(i, pred[i], test_labels, test_images)
plt.subplot(2,2,4)
plot_value_array(i, pred[i], test_labels)

plt.tight_layout()


# Same as Above but with 40 Images

# In[ ]:


num_rows = 8
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(4, 4))
for i in range(num_images):
    plt.subplot(2, 21, i + 1)
    plot_image(i, pred[i], test_labels, test_images)
    plt.subplot(num_rows, 21, i+2)
    plot_value_array(i, pred[i], test_labels)

plt.tight_layout()

