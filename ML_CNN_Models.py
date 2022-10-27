# Package import
from DataLoad import data_Load

import os
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler

import h5py

import warnings
warnings.filterwarnings('ignore')

# Loading data
X, Y = data_Load()

#------------------- CNN Model -------------------------

# Initializing label encoder for multi-class classification
lb=LabelEncoder()

# Fiting and transforming labels
Y1=np_utils.to_categorical(lb.fit_transform(Y))

# Train, test, and validation split
X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.3, random_state=1, stratify = Y,shuffle=True)

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=42,test_size=0.1,shuffle=True)

# Initializing scalar
scaler=StandardScaler()

# Fiting scalar on train, test and validation data
X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

X_val=scaler.transform(X_val)

X_train=np.expand_dims(X_train,axis=2)

X_val=np.expand_dims(X_val,axis=2)

X_test=np.expand_dims(X_test,axis=2)

# Parameters for CNN models
early_stop=EarlyStopping(monitor='val_accuracy',mode='auto',patience=5,restore_best_weights=True)

lr_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

# Initializing epoch and batch size
EPOCH=10

BATCH_SIZE=64

# CNN models architecture
cnn_model=tf.keras.Sequential([
    
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    
    L.Flatten(),
    L.Dense(512,activation='relu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])
cnn_model_tuned=tf.keras.Sequential([
    
    L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='gelu',input_shape=(X_train.shape[1],1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='gelu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    
    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='gelu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    
    
    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='gelu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    
    L.Flatten(),
    L.Dense(512,activation='gelu'),
    L.BatchNormalization(),
    L.Dense(7,activation='softmax')
])

# Compiling different CNN models.
cnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

cnn_model_tuned.compile(optimizer='adamax',loss='categorical_crossentropy',metrics='accuracy')

# Fiting and Saving CNN models.
history1=cnn_model.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_val,y_val), batch_size=BATCH_SIZE,callbacks=[early_stop,lr_reduction])

cnn_model.save(r'Speech-Emotion-Recognition-with-Audio/models/finalized_CNN_model.h5')

history2=cnn_model_tuned.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_val,y_val), batch_size=BATCH_SIZE,callbacks=[early_stop,lr_reduction])

cnn_model_tuned.save(r'Speech-Emotion-Recognition-with-Audio/models/finalized_CNN_tuned_model.h5')

# Saving the CNN models trained history.
hist1_df = pd.DataFrame(history1.history) 

hist2_df = pd.DataFrame(history2.history)  

# Converting models history to .CSV files.
hist1_csv_file = r'Speech-Emotion-Recognition-with-Audio/CNN Models Training History/history1.csv'

with open(hist1_csv_file, mode='w') as f1:
    
    hist1_df.to_csv(f1)
    
hist2_csv_file = r'Speech-Emotion-Recognition-with-Audio/CNN Models Training History/history2.csv'

with open(hist2_csv_file, mode='w') as f2:
    
    hist2_df.to_csv(f2)






