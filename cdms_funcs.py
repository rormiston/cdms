import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io as sio
import numpy as np
import json


def filter_data(dataset):
    b, a = sig.ellip(4, 0.01, 120, 0.125)
    for event in range(dataset.shape[0]):
        for detector in range(dataset.shape[2]):
            filtered = dataset[event, :, detector]
            fgust = sig.filtfilt(b, a, filtered, method="gust")
            dataset[event, :, detector] = fgust

    return dataset


def prepare_cdms(dataset):
    dd = {}
    for event in range(dataset.shape[0]):
        mn = np.min(dataset[event, :, :])
        mx = np.max(dataset[event, :, :])
        dd[event] = {'min':mn, 'max':mx}
        dataset[event, :, :] = (dataset[event, :, :] - mn) / (mx - mn)

    b, a = sig.ellip(4, 0.01, 120, 0.125)
    for event in range(dataset.shape[0]):
        for detector in range(dataset.shape[2]):
            filtered = dataset[event, :, detector]
            fgust = sig.filtfilt(b, a, filtered, method="gust")
            dataset[event, :, detector] = fgust

    return dataset, dd


def set_plot_style():
    plt.style.use('bmh')
    matplotlib.rcParams.update({
        'axes.grid': True,
        'axes.titlesize': 'medium',
        'font.family': 'serif',
        'font.size': 12,
        'grid.color': 'w',
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.linewidth': 1,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 13,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.1,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'text.usetex': False,
        'text.latex.preamble': r'\usepackage{txfonts}'
    })

    matplotlib.rc("savefig", dpi=100)
    matplotlib.rc("figure", figsize=(7, 4))


def make_data(fname, full=False):
    data_file = open(fname)
    lib = json.load(data_file)
    data_file.close()
    data = lib["data"]
    keys = ['Energy'] + sorted([x for x in data[0].keys() if x.startswith('P')])

    if full:
        output = 'cdms_event_data.mat'
        full_array = np.zeros(shape=(len(data), len(data[0]['PB']), len(keys)))
        for event in range(full_array.shape[0]):
            for i in range(full_array.shape[2]):
                full_array[event, :, i] = data[event][keys[i]]

    else:
        output = 'short_data.mat'
        full_array = np.zeros(shape=(400, 2048, len(keys)))
        for event in range(400):
            for i in range(full_array.shape[2]):
                if isinstance(data[event][keys[i]], list):
                    full_array[event, :, i] = data[event][keys[i]][:2048]
                else:
                    full_array[event, :, i] = data[event][keys[i]]

    sio.savemat(output, {'data':full_array})
