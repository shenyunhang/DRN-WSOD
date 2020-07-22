#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import sys

from detectron.utils.io import save_object
from detectron.utils.io import load_object

if __name__ == '__main__':
    original_weights_file = sys.argv[1]
    file_out = sys.argv[2]

    out_blobs = {}

    used_blobs = set()

    original_src_blobs = load_object(original_weights_file)

    print('====================================')
    print('get params in original weights')
    for blob_name in sorted(original_src_blobs.keys()):
        if 'bn_s' in blob_name:
            pass
        else:
            continue

        bn_name = blob_name.rsplit('_', 1)[0]
        conv_name = blob_name.rsplit('_', 2)[0]
        if 'res_conv1_bn_s' in blob_name:
            conv_name = 'conv1'

        bn_s_name = blob_name
        bn_b_name = bn_name + '_b'

        conv_w_name = conv_name + '_w'
        conv_b_name = conv_name + '_b'

        print(blob_name, bn_name, bn_s_name, bn_b_name, conv_name, conv_w_name,
              conv_b_name)

        if conv_b_name not in original_src_blobs.keys():
            original_src_blobs[conv_b_name] = np.zeros(
                (original_src_blobs[conv_w_name].shape[0]),
                dtype=original_src_blobs[conv_w_name].dtype)

        print(original_src_blobs[bn_s_name].shape,
              original_src_blobs[bn_b_name].shape,
              original_src_blobs[conv_w_name].shape,
              original_src_blobs[conv_b_name].shape)

        c_out, c_in, k_h, k_w = original_src_blobs[conv_w_name].shape

        # bn_w_tile = np.tile(original_src_blobs[bn_s_name],(1,c_in,k_h,k_w))

        out_blobs[conv_w_name] = original_src_blobs[conv_w_name] * np.tile(
            np.reshape(original_src_blobs[bn_s_name], (c_out, 1, 1, 1)),
            (1, c_in, k_h, k_w))

        out_blobs[conv_b_name] = original_src_blobs[
            conv_b_name] * original_src_blobs[bn_s_name] + original_src_blobs[
                bn_b_name]

        used_blobs.add(bn_s_name)
        used_blobs.add(bn_b_name)
        used_blobs.add(conv_w_name)
        used_blobs.add(conv_b_name)

    for blob_name in sorted(original_src_blobs.keys()):
        if blob_name in used_blobs:
            continue
        out_blobs[blob_name] = original_src_blobs[blob_name]

    print('Wrote blobs:')
    print(sorted(out_blobs.keys()))

    print(len(original_src_blobs))
    print(len(out_blobs))

    save_object(out_blobs, file_out)
