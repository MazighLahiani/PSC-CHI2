
import joblib
from rdkit.Chem import Draw
from rdkit.Chem import AllChem as Chem
import gzip
import os
import sys
Type = sys.getfilesystemencoding()

from sklearn.ensemble import RandomForestClassifier
from syba.syba import SybaClassifier, SmiMolSupplier
from datetime import datetime
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import optimizers

import numpy as np

def create_my_model5():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=10))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model

def create_my_model():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=1))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model

def create_my_model7():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=1))
    model.add(layers.LSTM(264))
    model.add(layers.Dense(264))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model

def create_my_model9():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=10))
    model.add(layers.LSTM(264))
    model.add(layers.Dense(264))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model


def create_my_model10():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=100))
    model.add(layers.LSTM(264))
    model.add(layers.Dense(264))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model

def create_my_model11():
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1024, output_dim=264))
    model.add(layers.LSTM(264))
    model.add(layers.Dense(264))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['mae', 'accuracy'])
    return model


model = create_my_model11()
# How to load a pre-trained model 
# model.load_weights('./data/rnn_lstm_V8.1000.10_weights.h5')

nBits = 1024
syn_fps = np.array([np.array(Chem.GetMorganFingerprintAsBitVect(spls[0],2,nBits=nBits)) for spls in SmiMolSupplier(gzip.open("./data/structures_2.csv.gz", mode="rt"), header=True, smi_col=1)])
n = len(syn_fps)
syn_classes = np.ones(n)

pour = 1
x_train_1 = syn_fps[:int(n*0.8)]
y_train_1 = syn_classes[:int(n*0.8)]
x_test_1 = syn_fps[int(n*0.8):]  
y_test_1 = syn_classes[int(n*0.8):] 

non_fps = [np.array(Chem.GetMorganFingerprintAsBitVect(spls[0],2,nBits=nBits)) for spls in SmiMolSupplier(gzip.open("./data/structures_1.csv.gz", mode="rt"), header=True, smi_col=2)]
n = len(non_fps)
non_classes = np.zeros(n)

x_train_0 = non_fps[:int(n*0.8)]
y_train_0 = non_classes[:int(n*0.8)]
x_test_0 = non_fps[int(n*0.8):] 
y_test_0 = non_classes[int(n*0.8):]


x_train = np.concatenate((x_train_1, x_train_0))
y_train = np.concatenate((y_train_1, y_train_0))
x_test = np.concatenate((x_test_1, x_test_0))
y_test = np.concatenate((y_test_1, y_test_0))

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=1000, epochs=10)


os.system('echo dans le python')
os.system('pwd')
os.system('ls -ls')
model.save('./out/')
model.save_weights('/scratch/lahiani/lstm/out/rnn_lstm_V11.1000.10_weights.h5')
joblib.dump(history.history, "./out/rnn_lstm_V11.1000.10_weights.history")



