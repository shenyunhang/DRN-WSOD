#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2
import os
import sys
import re

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import load_object
import detectron.utils.vis as vis_utils

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

import matplotlib.pyplot as plt
import sys
import subprocess
import numpy as np

print(sys.argv)

SNAPSHOT_EPOCHS = 2
LOG_PERIOD = 320


def get_loss():
    log_path = sys.argv[1]
    log_file = open(log_path, "r")
    prefix_path = os.path.splitext(log_path)[0]

    loss_values = []
    iter_values = []
    snapshot_values = []

    it = 0
    for line in log_file.readlines():
        line = line.strip()

        ma = re.search('model_final\.pkl', line)
        if ma is not None:
            break

        ma = re.search('model_iter[0-9]+[.]pkl', line)
        if ma is not None:
            ma = re.search('[0-9]+', ma.group())
            snapshot_value = float(ma.group())
            snapshot_values.append(snapshot_value)

        ma = re.search('"loss": "[0-9]+([.][0-9]+)?"', line)
        if ma is None:
            continue
        # print(ma)
        # print(ma.group())
        # print(ma.groups())

        ma = re.search('[0-9]+([.][0-9]+)?', ma.group())
        loss_value = float(ma.group())
        loss_values.append(loss_value)

        iter_values.append(it)
        it += LOG_PERIOD

    return loss_values, iter_values, snapshot_values


def get_mAP():
    log_path = sys.argv[2]
    log_file = open(log_path, "r")
    prefix_path = os.path.splitext(log_path)[0]

    mAP_values = []

    for line in log_file.readlines():
        line = line.strip()
        ma = re.search('Average Precision  \(AP\) @\[ IoU=0\.50:0\.95 \| area=   all \| maxDets=100 \] = [0-9]+([.][0-9]+)?', line)
        if ma is None:
            continue
        # print(ma)
        # print(ma.group())
        # print(ma.groups())

        ma = re.search(' = [0-9]+([.][0-9]+)?', ma.group())
        ma = re.search('[0-9]+([.][0-9]+)?', ma.group())
        mAP_value = float(ma.group())
        mAP_values.append(mAP_value)

    # test in reverse order
    return mAP_values[0], mAP_values[4:][::-1]


def draw():
    loss_values, iter_values, snapshot_value = get_loss()
    map_final, mAP_values = get_mAP()
    mAP_epoch_values = range(SNAPSHOT_EPOCHS,
                             SNAPSHOT_EPOCHS * len(mAP_values) + 1,
                             SNAPSHOT_EPOCHS)

    # scale epoch
    scale_epoch_values = [
        i * mAP_epoch_values[-1] / len(loss_values)
        for i in range(len(loss_values))
    ]

    log_path = sys.argv[1]
    uuid = os.path.basename(log_path).split(' ')[0]
    output_dir = os.path.join(os.path.dirname(log_path), 'draw')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(uuid)

    # plot the data
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(scale_epoch_values, loss_values, 'r', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(mAP_epoch_values, mAP_values, 'go', linewidth=1.5)
    ax2.set_ylim([0.0, 0.2])

    major_ticks = np.arange(0, mAP_epoch_values[-1], 5)
    # minor_ticks = np.arange(0, mAP_epoch_values[-1], 1)
    ax1.set_xticks(major_ticks)
    # ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='both')

    major_ticks = np.arange(0.0, 0.2, 0.05)
    # minor_ticks = np.arange(40, 90, 1)
    ax2.set_yticks(major_ticks)
    # ax2.set_yticks(minor_ticks, minor=True)
    ax2.grid(which='both')

    # ax2.grid(which='minor', alpha=0.2)
    # ax2.grid(which='major', alpha=0.5)

    # set the limits
    ax1.set_xlim([0, mAP_epoch_values[-1]])
    ax1.set_ylim([0, loss_values[0]])
    plt.savefig(os.path.join(output_dir, uuid + '_plot.png'), dpi=1200)

    # set the limits
    ax1.set_xlim([0, mAP_epoch_values[-1]])
    ax1.set_ylim([0, 8])
    plt.savefig(os.path.join(output_dir, uuid + '_plot_8.png'), dpi=1200)

    # set the limits
    ax1.set_xlim([0, mAP_epoch_values[-1]])
    ax1.set_ylim([0, 6])
    plt.savefig(os.path.join(output_dir, uuid + '_plot_6.png'), dpi=1200)


draw()
