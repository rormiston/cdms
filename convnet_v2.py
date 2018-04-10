from __future__ import print_function
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(3301)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import scipy.io as sio


# Set some global params
batch_size = 16
num_classes = 1
epochs = 500
datafile = 'mixed_data_with_shifted_cal_wide.mat'
plotDir = 'WideNormedCal'

# Shape data for CNN
full_data = sio.loadmat(datafile)['data']
t_frac  = int(0.8 * full_data.shape[0])
x_train = full_data[:t_frac, :, 1:]
y_train = full_data[:t_frac, 0, 0]
x_test  = full_data[t_frac:, :, 1:]
y_test  = full_data[t_frac:, 0, 0]

img_rows, img_cols = x_train.shape[1], x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Run the network
model = Sequential()
model.add(Conv2D(8, kernel_size=(16, 3),
                 activation='relu',
                 input_shape=input_shape, padding='same'))
model.add(Conv2D(8, (16, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='linear'))

model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

# Start plotting things
if not os.path.isdir(plotDir):
    os.system('mkdir -p {}'.format(plotDir))

yhat = model.predict(x_test)
plt.plot(yhat, label='pred')
plt.plot(y_test, label='tar')
plt.title('Prediction vs Target')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('keV')
plt.savefig('{}/keras_cal_dist.png'.format(plotDir))
plt.close()

bins = bins=np.arange(9.5, 11.83, 0.03)
plt.hist(y_test, label='target', bins=bins, alpha=0.5)
plt.hist(yhat, label='prediction', bins=bins, alpha=0.5)
plt.legend(loc='upper right')
plt.title('Event Energy Distribution')
plt.xlabel('Energy (keV)')
plt.ylabel('# Events')
plt.savefig('{}/keras_cal_hist.png'.format(plotDir))
plt.close()

plt.plot(history.history['loss'][5:], label='loss')
plt.plot(history.history['val_loss'][5:], label='val loss')
plt.title('Testing/Training Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('{}/keras_loss.png'.format(plotDir))
plt.close()

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.title('Testing/Training Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('{}/keras_accuracy.png'.format(plotDir))
plt.close()
