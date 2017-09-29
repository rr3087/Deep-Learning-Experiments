#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:12:02 2017

@author: rrana
"""
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Activation, LSTM, GRU, Dropout
from keras.layers.normalization import BatchNormalization
import numpy
from sklearn.model_selection import train_test_split # fix random seed for reproducibility
from sklearn.metrics import r2_score

numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("tjx_data.csv", delimiter=",", skiprows = 1)
# split into input (X) and output (Y) variables
X = dataset[:,0:47]
y = dataset[:,47]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state=40)
Xtrain=Xtrain.reshape(Xtrain.shape[0], 47, 1)
Xtest=Xtest.reshape(Xtest.shape[0], 47, 1)
#==============================================================================
# Xtrain = Xtrain_.T
# ytrain = ytrain_.T
# Xtest = Xtest_.T
# ytest = ytest_.T
#==============================================================================
# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# model.add(Dense(8, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, nb_epoch=150, batch_size=10,  verbose=2)
# # calculate predictions
# predictions = model.predict(X)
# # round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

model = Sequential()
model.add(Conv1D(1024, kernel_size=3, padding='same', input_shape=(47,1), activation='tanh'))
model.add(Dropout(0.3))
model.add(Conv1D(512, kernel_size=3, padding='same', activation='tanh'))
model.add(Dropout(0.3))
model.add(Conv1D(256, kernel_size=3, padding='same', activation='tanh'))
model.add(Dropout(0.3))
# model.add(GRU(1024, input_shape=(47,1), activation='relu', return_sequences=True))
#model.add(BatchNormalization())
# model.add(Activation('relu'))
#model.add(Conv1D(47, kernel_size=3, padding='same'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
#model.add(Conv1D(23, kernel_size=3, padding='same', activation='relu'))
# model.add(Dense(2048, input_shape=(47,1), activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))

# #==============================================================================
# # model.add(Dense(12, init='uniform', activation='relu'))
# # model.add(Dense(1, init='uniform', activation='relu'))
# #==============================================================================
model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', 
              optimizer=keras.optimizers.Adadelta()
              )

model.fit(Xtrain, ytrain, epochs=400, batch_size=100, validation_data=(Xtest, ytest), verbose=1)

print "testing r_square", r2_score(ytest ,model.predict(Xtest))
print "training r_square", r2_score(ytrain, model.predict(Xtrain))


