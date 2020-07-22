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

if __name__ == '__main__':

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    pkl_data = load_object(in_path)

    pkl_data = pkl_data['blobs']
    keys = pkl_data.keys()

    for k in list(keys):
        if 'momentum' in k:
            print('delete ', k)
            t = pkl_data.pop(k, None)

    save_object(pkl_data, out_path)
