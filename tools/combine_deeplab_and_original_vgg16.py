"""Script to convert Mutiscale Combinatorial Grouping proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import numpy as np
import scipy.io as sio
import sys
import os

from detectron.utils.io import save_object

if __name__ == '__main__':
    original_weights_file = sys.argv[1]
    deeplab_weights_file = sys.argv[2]
    file_out = sys.argv[3]

    out_blobs = {}

    with open(deeplab_weights_file, 'r') as f:
        deeplab_src_blobs = pickle.load(f)

    with open(original_weights_file, 'r') as f:
        original_src_blobs = pickle.load(f)

    print('====================================')
    print('get params in original VGG16')
    for blob_name in sorted(original_src_blobs.keys()):
        print(blob_name)
        if 'fc8' in blob_name:
            print('fc8 layer not saved')
            continue
        out_blobs[blob_name] = original_src_blobs[blob_name]

    print('====================================')
    print('get params in deeplab VGG16')
    for blob_name in sorted(deeplab_src_blobs.keys()):
        if blob_name in original_src_blobs.keys():
            print('check param in two weight: ', blob_name)
            deeplab_src_blob = deeplab_src_blobs[blob_name]
            original_src_blob = original_src_blobs[blob_name]
            assert deeplab_src_blob.shape == original_src_blob.shape
            assert deeplab_src_blob.sum() == original_src_blob.sum()
            continue

        print(blob_name)
        if 'fc8' in blob_name:
            print('fc8 layer not saved')
            continue
        out_blobs[blob_name] = deeplab_src_blobs[blob_name]

    print('Wrote blobs:')
    print(sorted(out_blobs.keys()))

    save_object(out_blobs, file_out)
