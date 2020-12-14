# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:21:40 2020

@author: Evgeniy
"""
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import pandas as pd


def get_images(path):
    filelist = os.listdir(path)
    for fname in filelist:
        temp_img = Image.open(path+fname)
        temp_img = ImageOps.grayscale(temp_img)
        #temp_img = temp_img.convert("RGB")
        temp_img = temp_img.resize((28,28))
        temp_img = ImageOps.invert(temp_img)
        x = np.array([np.array(temp_img) for fname in filelist])
    return x

def prepare_data():
    x_train = get_images('./datasets/dataset-wb-100x100/train/0/')
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/1/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/2/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/3/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/4/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/5/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/6/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/7/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/8/')), axis=0)
    x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/9/')), axis=0)

    x_train_flat = x_train.reshape(-1, 28*28).astype(float)
    x_train_float = x_train_flat.astype(np.float) / 255 - 0.5

    y_train = pd.read_csv('./datasets/dataset-wb-100x100/labelsTrain.csv')
    y_train = y_train['label']

    x_test = get_images('./datasets/dataset-wb-100x100/validation/0/')
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/1/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/2/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/3/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/4/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/5/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/6/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/7/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/8/')), axis=0)
    x_test = np.concatenate((x_test, get_images('./datasets/dataset-wb-100x100/validation/9/')), axis=0)
    x_test_flat = x_test.reshape(-1, 28*28).astype(float)
    x_test_float = x_test_flat.astype(np.float) / 255 - 0.5

    y_test = pd.read_csv('./datasets/dataset-wb-100x100/labelsValidation.csv')
    y_test = y_test['label']

    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return x_train_float,y_train_oh,x_test_float,y_test_oh

def percp_model():
    model = Sequential()
    model.add(Dense(4,activation='relu',input_shape=(x_train_float.shape[1],)))
    model.add(Dense(4,activation='relu',))
    model.add(Dense(4,activation='relu',)) 
    model.add(Dense(10, activation='softmax',)) # первый скрытый слой
    return model

def make_default_model():
    model = Sequential()
    model.add(L.Conv2D(32, kernel_size=3,activation='relu', strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(L.Conv2D(64,kernel_size=3,activation='relu', strides=1, padding='same'))
    model.add(L.MaxPooling2D())
    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(64,kernel_size=3,activation='relu', strides=1, padding='same'))
    model.add(L.Conv2D(128,kernel_size=3,activation='relu', strides=1, padding='same'))
    model.add(L.MaxPooling2D())
    model.add(L.Dropout(0.25))
    model.add(L.Flatten()) 
    model.add(L.Dense(256,activation='elu'))
    model.add(L.Dropout(0.5))
    model.add(L.Dense(10,activation='softmax'))
    return model

def make_bn_model():
    model = Sequential()
    model.add(L.Conv2D(16, kernel_size=3,activation='sigmoidal', strides=1, padding='same', input_shape=(32, 32, 3)))
    model.add(L.Conv2D(32,kernel_size=3,activation='sigmoidal', strides=1, padding='same'))
    model.add(L.MaxPooling2D())
    model.add(L.Dropout(0.25))
    model.add(L.Conv2D(32,kernel_size=3,activation='sigmoidal', strides=1, padding='same'))
    model.add(L.Conv2D(64,kernel_size=3,activation='sigmoidal', strides=1, padding='same'))
    model.add(L.MaxPooling2D())
    model.add(L.Dropout(0.25))
    model.add(L.Flatten()) 
    model.add(L.Dense(256,activation='elu'))
    model.add(L.Dropout(0.5))
    model.add(L.Dense(10,activation='softmax'))
    return model



def train_model(make_model_func=make_bn_model, optimizer="adam"):
  BATCH_SIZE = 1
  EPOCHS = 10

  K.clear_session()
  model = make_model_func()

  model.compile(
      loss='categorical_crossentropy',
      optimizer=optimizer,
      metrics=['accuracy']
  )

  model.fit(
      x_train_float, y_train_oh,  # нормализованные данные
      batch_size=BATCH_SIZE,
      epochs=EPOCHS,
      validation_data=(x_test_float, y_test_oh),
      shuffle=True
  )
  
  return model

x_train_float,y_train_oh,x_test_float,y_test_oh = prepare_data()
train_model(make_model_func= percp_model)
"""
x_train = get_images('./datasets/dataset-wb-100x100/train/0/')
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/1/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/2/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/3/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/4/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/5/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/6/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/7/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/8/')), axis=0)
x_train = np.concatenate((x_train, get_images('./datasets/dataset-wb-100x100/train/9/')), axis=0)

x_train_flat = x_train.reshape(-1, 28*28).astype(float)
x_train_float = x_train_flat.astype(np.float) / 255 - 0.5

y_train = pd.read_csv('./datasets/dataset-wb-100x100/labelsTrain.csv')
y_train = y_train['label']
"""