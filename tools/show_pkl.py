"""Script to convert Mutiscale Combinatorial Grouping proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os

from detectron.utils.io import save_object

if __name__ == '__main__':
    weights_file = sys.argv[1]

    with open(weights_file, 'r') as f:
        src_blobs = pickle.load(f)

    print('====================================')
    print('get params')
    for blob_name in sorted(src_blobs.keys()):
        blob = src_blobs[blob_name]
        print(blob_name, blob.shape, blob.max(), blob.min(), blob.mean())
