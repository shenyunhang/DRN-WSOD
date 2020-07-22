from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import argparse
import copy
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os
import pprint
import sys

if __name__ == '__main__':
    weights_file = sys.argv[1]
    with open(weights_file, 'r') as f:
        src_blobs = pickle.load(f)

    for key in sorted(src_blobs.keys()):
        src_blob = src_blobs[key]
        print(key, src_blob.shape)
