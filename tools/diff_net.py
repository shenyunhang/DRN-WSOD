from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os
from time import strftime

from detectron.datasets.json_dataset_wsl import JsonDataset

import matplotlib as mpl
import matplotlib.pyplot as plt

from detectron.utils.io import save_object
from detectron.utils.io import load_object

if __name__ == '__main__':
    net_1 = sys.argv[1]
    net_2 = sys.argv[2]

    pkl_1 = load_object(net_1)
    pkl_2 = load_object(net_2)

    if 'blobs' in pkl_1.keys():
        pkl_1 = pkl_1['blobs']
    if 'blobs' in pkl_2.keys():
        pkl_2 = pkl_2['blobs']

    print(pkl_1.keys())
    print(pkl_2.keys())

    for key in sorted(pkl_1.keys()):
        if 'momentum' in key:
            continue

        if key.endswith('_b'):
            continue

        if '_bn_' in key:
            continue

        if key in pkl_2.keys():
            pass
        else:
            continue

        diff = pkl_1[key] - pkl_2[key]
        normF = np.linalg.norm(diff)

        diff = pkl_1[key] - pkl_2[key]
        diff = diff * diff
        norm2 = np.mean(np.sqrt(diff))

        diff = np.abs(pkl_1[key] - pkl_2[key]) / np.abs(pkl_1[key])
        diffR = np.mean(diff)

        diff = np.abs(pkl_1[key] - pkl_2[key])
        diffA = np.mean(diff)

        print(key, np.mean(np.abs(pkl_1[key])), np.mean(np.abs(pkl_2[key])),
              ' normF: ', normF, ' norm2: ', norm2, ' diffR: ', diffR,
              ' diffA: ', diffA, np.corrcoef(pkl_1[key].flat, pkl_2[key].flat))
