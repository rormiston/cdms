import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
np.random.seed(3301)

from keras.models import Sequential
from keras.layers import Dense, Dropout


# global params
activation = 'linear'
batch_size = 10
epochs     = 500
verbose    = 2
optimizer  = 'adam'

# get data
mat_file = sio.loadmat('short_data.mat')
# mat_file = sio.loadmat('medium_data.mat')
data = mat_file['data']

# split into training and testing
tfrac   = int(data.shape[0] * 0.8)
x_train = data[:tfrac, :, 1:]
y_train = data[:tfrac, 0, 0]
x_test  = data[tfrac:, :, 1:]
y_test  = data[tfrac:, 0, 0]

# reshape for network
y_train = y_train.reshape(len(y_train))
y_test  = y_test.reshape(len(y_test))
temp    = np.zeros((tfrac, x_train.shape[1] * 5))
ts_len  = x_train.shape[1]

for j in range(tfrac):
    for i in range(5):
        temp[j, ts_len * i: ts_len * (i + 1)] = x_train[j, :, i]
x_train = temp

temp = np.zeros((x_test.shape[0], x_test.shape[1] * 5))
for j in range(x_test.shape[0]):
    for i in range(5):
        temp[j, ts_len * i: ts_len * (i + 1)] = x_test[j, :, i]
x_test = temp

# build network
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation=activation))
model.add(Dense(64, activation=activation))
model.add(Dense(32, activation=activation))
model.add(Dense(16, activation=activation))
model.add(Dense(8, activation=activation))
model.add(Dense(1))

# compile and fit
model.compile(loss='mae', optimizer=optimizer)
history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs     = epochs,
                    validation_data = (x_test, y_test),
                    verbose    = verbose)

# make a prediction
yhat = model.predict(x_test)
yhat = yhat.reshape(len(yhat))

# show results
bins = np.arange(0, 41, 1)
plt.hist(yhat, label='prediction', bins=bins, alpha=0.8)
plt.hist(y_test, label='target', bins=bins, alpha=0.8)
plt.ylabel('Count')
plt.xlabel('Energy (keV)')
plt.title('Network Prediction vs. Target Energy Histogram')
plt.legend()
plt.xlim([0, 20])
plt.savefig('dense_hist.png')
plt.close()

plt.plot(yhat, label='prediction')
plt.plot(y_test, label='target')
plt.xlabel('Event')
plt.ylabel('Energy (keV)')
plt.title('Network Prediction vs. Target Energy Distribution')
plt.legend()
plt.savefig('dense_dist.png')
plt.close()

plt.plot(history.history['loss'][3:], label='training loss')
plt.plot(history.history['val_loss'][3:], label='testing loss')
plt.legend()
plt.title('Dense Network Loss')
plt.savefig('dense_loss.png')
plt.close()
