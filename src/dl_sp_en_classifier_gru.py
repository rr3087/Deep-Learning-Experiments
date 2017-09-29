fsd## run only for 76 epochs, don't run the 3rd ls assignment.

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Activation, LSTM, GRU, Dropout, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
import numpy as np
from scipy.io import mmread
from scipy import sparse
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split # fix random seed for reproducibility
from sklearn.metrics import r2_score

X = np.loadtxt("sp_en_vectors2.csv", dtype='int', delimiter=',')
y = np.loadtxt("wgs_500k_en_labels.csv", dtype='int', usecols=(0,))
y = to_categorical(y-1, nb_classes=3)

print X.shape

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=40)
Xtrain=Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
Xtest=Xtest.reshape(Xtest.shape[0], 1,  Xtest.shape[1])


model = Sequential()

#model.add(Conv1D(1024, 5, input_shape=(2401, 1), activation='relu'))
#model.add(Conv1D(1024, 5, activation='relu'))
#model.add(Conv1D(1024, 3, activation='relu'))
#model.add(Conv1D(512, 3, activation='relu'))
#model.add(Conv1D(512, 3, activation='relu')) 

model.add(GRU(3600, input_shape=(1,2401), 
		activation='tanh', 
		return_sequences=True))
#model.add(BatchNormalization())
#model.add(Activation('tanh'))
model.add(Dropout(0.9))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
#model.add(Flatten())
#model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
				optimizer = keras.optimizers.Adadelta(),
				metrics=['accuracy'])

model.fit(Xtrain, ytrain,
			nb_epoch=60,
			batch_size=1000,
			validation_data=(Xtest, ytest),
			verbose = 1)

#model.save('best_model2_5.h5')

model.optimizer.lr.assign(2)

model.fit(Xtrain, ytrain,
		nb_epoch=20, 
		batch_size=1000,
		validation_data=(Xtest, ytest),
		verbose=1)

model.optimizer.lr.assign(3)

model.fit(Xtrain, ytrain, 
		nb_epoch=10, 
		batch_size=1000,
		validation_data=(Xtest, ytest),
		verbose=1)

model.save('best_model2_5.h5')


#y_train_pred = model.predict(Xtrain)
#y_test_pred = model.predict(Xtest)
#y_train_analyzed = np.hstack((y_train_pred, ytrain))
#y_test_analyzed = np.hstack((y_test_pred, ytest))
#np.savetxt('train_analysis_labels.csv', y_train_analyzed, delimiter=',')
#np.savetxt('Xtrain.csv', Xtrain,  delimiter=',')
#np.savetxt('test_analysis_labels.csv', y_test_analyzed, delimiter=',')
#np.savetxt('Xtest.csv', Xtest, dtype delimiter=',')



