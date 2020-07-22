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
from collections import OrderedDict

from six.moves import cPickle as pickle
from detectron.utils.io import save_object
from detectron.utils.io import load_object

import torch
from torch.utils import model_zoo
import torchvision

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

c_2_p = {
    'vgg16': [
        ['conv1_1', 'features.0'],
        ['conv1_2', 'features.2'],
        ['conv2_1', 'features.5'],
        ['conv2_2', 'features.7'],
        ['conv3_1', 'features.10'],
        ['conv3_2', 'features.12'],
        ['conv3_3', 'features.14'],
        ['conv4_1', 'features.17'],
        ['conv4_2', 'features.19'],
        ['conv4_3', 'features.21'],
        ['conv5_1', 'features.24'],
        ['conv5_2', 'features.26'],
        ['conv5_3', 'features.28'],
        ['fc6', 'classifier.0'],
        ['fc7', 'classifier.3'],
        ['fc8', 'classifier.6'],
    ],
    'vgg19': [
        ['conv1_1', 'features.0'],
        ['conv1_2', 'features.2'],
        ['conv2_1', 'features.5'],
        ['conv2_2', 'features.7'],
        ['conv3_1', 'features.10'],
        ['conv3_2', 'features.12'],
        ['conv3_3', 'features.14'],
        ['conv3_4', 'features.16'],
        ['conv4_1', 'features.19'],
        ['conv4_2', 'features.21'],
        ['conv4_3', 'features.23'],
        ['conv4_4', 'features.25'],
        ['conv5_1', 'features.28'],
        ['conv5_2', 'features.30'],
        ['conv5_3', 'features.32'],
        ['conv5_4', 'features.34'],
        ['fc6', 'classifier.0'],
        ['fc7', 'classifier.3'],
        ['fc8', 'classifier.6'],
    ],
    'vgg16_bn': [
        ['conv1_1', 'features.0'],
        ['conv1_2', 'features.3'],
        ['conv2_1', 'features.7'],
        ['conv2_2', 'features.10'],
        ['conv3_1', 'features.14'],
        ['conv3_2', 'features.17'],
        ['conv3_3', 'features.20'],
        ['conv4_1', 'features.24'],
        ['conv4_2', 'features.27'],
        ['conv4_3', 'features.30'],
        ['conv5_1', 'features.34'],
        ['conv5_2', 'features.37'],
        ['conv5_3', 'features.40'],
        ['fc6', 'classifier.0'],
        ['fc7', 'classifier.3'],
        ['fc8', 'classifier.6'],
    ],
    'vgg19_bn': [
        ['conv1_1', 'features.0'],
        ['conv1_2', 'features.3'],
        ['conv2_1', 'features.7'],
        ['conv2_2', 'features.10'],
        ['conv3_1', 'features.14'],
        ['conv3_2', 'features.17'],
        ['conv3_3', 'features.20'],
        ['conv3_4', 'features.23'],
        ['conv4_1', 'features.27'],
        ['conv4_2', 'features.30'],
        ['conv4_3', 'features.33'],
        ['conv4_4', 'features.36'],
        ['conv5_1', 'features.40'],
        ['conv5_2', 'features.43'],
        ['conv5_3', 'features.46'],
        ['conv5_4', 'features.49'],
        ['fc6', 'classifier.0'],
        ['fc7', 'classifier.3'],
        ['fc8', 'classifier.6'],
    ],
}

param = [['_w', '.weight'], ['_b', '.bias']]

if __name__ == '__main__':

    for model_name in c_2_p.keys():
        print('============================================================')
        print(model_name)
        print('============================================================')
        url = model_urls[model_name]

        checkpoint = model_zoo.load_url(url)

        print(checkpoint.keys())

        dict_data = checkpoint
        new_dict_data = OrderedDict()

        cnt = 0
        for name_map in c_2_p[model_name]:
            for param_map in param:
                old_key = name_map[1] + param_map[1]
                new_key = name_map[0] + param_map[0]

                new_dict_data[new_key] = dict_data[old_key].numpy()
                print(cnt, '\tmap: ', old_key, ' to ', new_key)
                cnt += 1

        cnt = 0
        if model_name.endswith('_bn'):
            for name_map in c_2_p[model_name]:
                if 'conv' in name_map[0]:
                    name_split = name_map[1].split('.')
                    name_split[1] = int(name_split[1]) + 1
                    name_map_bn = name_split[0] + '.' + str(name_split[1])

                    key_bn_w = name_map_bn + param[0][1]
                    key_bn_b = name_map_bn + param[1][1]
                    key_bn_m = name_map_bn + '.running_mean'
                    key_bn_v = name_map_bn + '.running_var'

                    key_conv_w = name_map[0] + param[0][0]
                    key_conv_b = name_map[0] + param[1][0]

                    # bn to aff
                    bn_scale = dict_data[key_bn_w].numpy()
                    bn_bias = dict_data[key_bn_b].numpy()
                    bn_mean = dict_data[key_bn_m].numpy()
                    bn_var = dict_data[key_bn_v].numpy()

                    bn_std = np.sqrt(bn_var + 1e-5)
                    aff_scale = bn_scale / bn_std
                    aff_bias = bn_bias - bn_mean * bn_scale / bn_std

                    # merge aff to conv
                    conv_w = new_dict_data[key_conv_w]
                    conv_b = new_dict_data[key_conv_b]
                    c_out, c_in, k_h, k_w = conv_w.shape

                    # bn_w_tile = np.tile(aff_scale, (1,c_in,k_h,k_w))

                    new_conv_w = conv_w * np.tile(
                        np.reshape(aff_scale, (c_out, 1, 1, 1)),
                        (1, c_in, k_h, k_w))

                    new_conv_b = conv_b * aff_scale + aff_bias

                    new_dict_data[key_conv_w] = new_conv_w
                    new_dict_data[key_conv_b] = new_conv_b

                    print(cnt, '\tmerge: ', name_map_bn, ' into ', name_map[0])
                    cnt += 1

        new_pkl_path = 'model/pytorch/' + model_name + '_pytorch.pkl'
        save_object(new_dict_data,
                    new_pkl_path,
                    pickle_format=pickle.HIGHEST_PROTOCOL)
