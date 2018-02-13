#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import cdms_lib

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


###################
# Set Environment #
###################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
cdms_lib.set_plot_style()

#################
# Global params #
#################
batch_size = 1
fs         = 512
epochs     = 1000
verbose    = 1
optimizer  = 'adam'

#############################################
# Get dataset, normalize filter and reshape #
#############################################
full_data = sio.loadmat('short_data.mat')['data']
prep_data, dd = cdms_lib.prepare_cdms(full_data, Filter=True)
cnn_data = prep_data.reshape(prep_data.shape[0],
                             prep_data.shape[1],
                             prep_data.shape[2], 1)

##################################
# Make training and testing data #
##################################
t_frac  = int(0.8 * cnn_data.shape[0])
x_train = cnn_data[:t_frac, :, 1:, :]
y_train = cnn_data[:t_frac, 0, 0 , :]
x_test  = cnn_data[t_frac:, :, 1:, :]
y_test  = cnn_data[t_frac:, 0, 0 , :]

# reshape the target data
y_train = y_train.reshape(y_train.shape[0], 1)
y_test  = y_test.reshape(y_test.shape[0], 1)

#####################
# Build the ConvNet #
#####################
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(16))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer=optimizer)
history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs     = epochs,
                    verbose    = verbose)

score = model.evaluate(x_test, y_test)
print('RMSE: {}'.format(score))

#############################
# Make prediction and Plots #
#############################
yhat = model.predict(x_test)
yhat = yhat.reshape(yhat.shape[0])

##############
# Make plots #
##############
y_test = y_test.reshape(len(y_test))
plt.plot(range(len(yhat)), y_test,  label='target')
plt.plot(range(len(yhat)), yhat, label='prediction')
plt.legend(loc='upper right')
plt.title('CDMS Energy Prediction')
plt.xlabel('Event')
plt.ylabel('Energy (keV)')
plt.savefig('energy_prediction.png')
plt.close()

plt.hist(y_test, label='target', alpha=0.5)
plt.hist(yhat, label='prediction', alpha=0.5)
plt.legend(loc='upper right')
plt.title('Event Energy Distribution')
plt.xlabel('Energy (keV)')
plt.ylabel('# Events')
plt.savefig('hist.png')
plt.close()

###########
# Rescale #
###########
for i in range(len(yhat)):
    mn = float(dd[x_train.shape[0] + i]['min'])
    mx = float(dd[x_train.shape[0] + i]['max'])
    yhat[i] = yhat[i] * (mx - mn) + mn

tar = sio.loadmat('short_data.mat')['data'][-len(yhat):, 0, 0]

#######################
# Make rescaled plots #
#######################
plt.plot(range(len(tar)), tar,  label='target')
plt.plot(range(len(tar)), yhat, label='prediction')
plt.legend(loc='upper right')
plt.title('CDMS Energy Prediction')
plt.xlabel('Event')
plt.ylabel('Energy (keV)')
plt.savefig('rescale_energy_prediction.png')
plt.close()

plt.hist(tar, label='target', alpha=0.5)
plt.hist(yhat, label='prediction', alpha=0.5)
plt.legend(loc='upper right')
plt.title('Event Energy Distribution')
plt.xlabel('Energy (keV)')
plt.ylabel('# Events')
plt.savefig('rescale_hist.png')
plt.close()
