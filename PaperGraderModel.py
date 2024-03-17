#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np


# In[3]:


print("TensorFlow version:", tf.__version__)


# In[4]:


#using handwritten dataset from Keras to train model
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[5]:


X_train_reshaped = np.expand_dims(X_train, axis=-1)


# In[6]:


data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(28,
                                                              28,
                                                              1)),
    layers.experimental.preprocessing.RandomRotation(0.25),
    layers.experimental.preprocessing.RandomZoom(0.25),
  ]
)


# In[7]:


model1 = models.Sequential([
    layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=100, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    #layers.Conv2D(filters=150, kernel_size=(3, 3), activation='relu'),
    #layers.MaxPooling2D((2, 2)),

    layers.Dropout(.25),

    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.fit(X_train_reshaped/255, y_train, epochs=5)


# In[8]:


model1.evaluate(X_test/255, y_test)


# In[9]:


#model1.save('model1.keras')

