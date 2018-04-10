import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.io as sio


#Load the library
fname = 'goodEvents.json'
data_file = open(fname)
lib = json.load(data_file)
data_file.close()
data = lib["data"]

keys = ['Energy'] + sorted([x for x in data[0].keys() if x.startswith('P')])
full_array = np.zeros(shape=(len(data), len(data[0]['PB']), len(keys)))


# for event in range(full_array.shape[0]):
#     for i in range(full_array.shape[2]):
#         full_array[event, :, i] = data[event][keys[i]]

# sio.savemat('cdms_event_data.mat', {'data':full_array})

# full_array = np.zeros(shape=(1000, 2048, len(keys)))
# for event in range(1000):
#     for i in range(full_array.shape[2]):
#         if isinstance(data[event][keys[i]], list):
#             full_array[event, :, i] = data[event][keys[i]][:2048]
#         else:
#             full_array[event, :, i] = data[event][keys[i]]

# sio.savemat('medium_data.mat', {'data':full_array})

# cal_array = np.zeros(shape=(400, len(data[0]['PB']), len(keys)))
# tot = 0
# for event in range(full_array.shape[0]):
#     energy = data[event]['Energy']
#     if np.abs(energy - 10) <= 0.5:
#         if tot < 400:
#             for i in range(cal_array.shape[2]):
#                 cal_array[tot, :, i] = data[event][keys[i]]
#                 if i == 0:
#                     cal_array[tot, :, i] = 10.0
#             tot += 1


# ###########################
# # 10 keV CALIBRATION DATA #
# ###########################
# cal_mean = 10.37
# tens = cal_mean * np.array([1 for _ in range(900)])
# ones = np.ones(900)
# rn = np.random.normal(0, 1, 900)
# rn /= np.max(rn)
# tens = tens + 0.1 * ones * rn

# cal_array = np.zeros(shape=(1500, len(data[0]['PB']), len(keys)))
# a, b = 0, 600
# ten_spot = 0

# for event in range(full_array.shape[0]):
#     energy = data[event]['Energy']
#     dt = np.abs(energy - cal_mean)
#     if dt >= 1.5:
#         if a < 600:
#             for i in range(full_array.shape[2]):
#                 cal_array[event, :, i] = data[event][keys[i]]
#             a += 1

# for event in range(full_array.shape[0]):
#     energy = data[event]['Energy']
#     dt = np.abs(energy - cal_mean)
#     if dt <= 0.5:
#         if 600 <= b < 1200:
#             for i in range(cal_array.shape[2]):
#                 if i == 0:
#                     cal_array[b, :, 0] = tens[ten_spot]
#                     ten_spot += 1
#                 else:
#                     cal_array[b, :, i] = data[event][keys[i]]
#             b += 1

#         if 1200 <= b < 1500:
#             for i in range(cal_array.shape[2]):
#                 if i == 0:
#                     cal_array[b, :, 0] = energy
#                 else:
#                     cal_array[b, :, i] = data[event][keys[i]]
#             b += 1

# low = cal_mean - 0.55
# high = cal_mean + 0.55
# plt.hist(cal_array[1200:, 0, 0], label='test', alpha=0.8, bins=np.arange(low, high, 0.05))
# plt.hist(cal_array[600:1200, 0, 0], label='train', alpha=0.8)
# plt.axvline(10.37, color='r', label='10.37keV')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events')
# plt.legend()
# plt.savefig('data_hist_train.png')
# plt.close()

# sio.savemat('mixed_data_with_shifted_cal_medium_normed.mat', {'data':cal_array})


# en_ar = []
# for event in range(full_array.shape[0]):
#     energy = data[event]['Energy']
#     en_ar.append(energy)

# b, v, _ = plt.hist(en_ar, label='raw data', bins=np.arange(0, 30, 0.1))
# vals = list(zip(b, v))
# vals = sorted(vals, key=lambda tup: tup[0])
# plt.axvline(10.37, color='r', label='10.37keV')
# plt.xlabel('Energy (keV)')
# plt.ylabel('Events')
# plt.legend()
# plt.savefig('data_hist_raw.png')
# plt.close()



################################
# Wide 10 keV CALIBRATION DATA #
################################
cal_mean = 10.37
tens = cal_mean * np.array([1 for _ in range(900)])
ones = np.ones(900)
rn = np.random.normal(0, 1, 900)
rn /= np.max(rn)
tens = tens + 0.25 * ones * rn

cal_array = np.zeros(shape=(1500, len(data[0]['PB']), len(keys)))
a, b = 0, 600
ten_spot = 0

for event in range(full_array.shape[0]):
    energy = data[event]['Energy']
    dt = np.abs(energy - cal_mean)
    if dt >= 4:
        if a < 600:
            for i in range(full_array.shape[2]):
                cal_array[event, :, i] = data[event][keys[i]]
            a += 1

for event in range(full_array.shape[0]):
    energy = data[event]['Energy']
    dt = np.abs(energy - cal_mean)
    if dt <= 2.5:
        if 600 <= b < 1200:
            for i in range(cal_array.shape[2]):
                if i == 0:
                    cal_array[b, :, 0] = tens[ten_spot]
                    ten_spot += 1
                else:
                    cal_array[b, :, i] = data[event][keys[i]]
            b += 1

        if 1200 <= b < 1500:
            for i in range(cal_array.shape[2]):
                if i == 0:
                    cal_array[b, :, 0] = energy
                else:
                    cal_array[b, :, i] = data[event][keys[i]]
            b += 1

low = cal_mean - 2.5
high = cal_mean + 2.6
plt.hist(cal_array[1200:, 0, 0], label='raw test (300)', alpha=0.8, bins=np.arange(low, high, 0.1))
# plt.hist(cal_array[600:1200, 0, 0], label='train', alpha=0.8)
plt.axvline(10.37, color='r', label='10.37keV')
plt.xlabel('Energy (keV)')
plt.ylabel('Events')
plt.legend()
plt.savefig('data_hist_wide.png')
plt.close()

sio.savemat('mixed_data_with_shifted_cal_wide.mat', {'data':cal_array})
