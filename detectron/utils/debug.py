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
"""Helper functions for working with Caffe2 networks (i.e., operator graphs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from caffe2.python import workspace
import detectron.utils.c2 as c2_utils

logger = logging.getLogger(__name__)


def print_net(model, namescope='gpu_0'):
    """Print the model network."""
    logger.info('Printing model: {}'.format(model.net.Name()))
    op_list = model.net.Proto().op
    for op in op_list:
        input_name = op.input
        # For simplicity: only print the first output
        # Not recommended if there are split layers
        output_name = str(op.output[0])
        op_type = op.type
        op_name = op.name

        if namescope is None or output_name.startswith(namescope):
            # Only print the forward pass network
            if output_name.find('grad') >= 0 or output_name.find('__m') >= 0:
                continue

            try:
                # Under some conditions (e.g., dynamic memory optimization)
                # it is possible that the network frees some blobs when they are
                # no longer needed. Handle this case...
                output_shape = workspace.FetchBlob(output_name).shape
            except BaseException:
                output_shape = '<unknown>'

            first_blob = True
            op_label = op_type + (op_name if op_name == '' else ':' + op_name)
            suffix = ' ------- (op: {})'.format(op_label)
            for j in range(len(input_name)):
                if input_name[j] in model.params:
                    continue
                input_blob = workspace.FetchBlob(input_name[j])
                if isinstance(input_blob, np.ndarray):
                    input_shape = input_blob.shape
                    logger.info('{:28s}: {:20s} => {:28s}: {:20s}{}'.format(
                        c2_utils.UnscopeName(str(input_name[j])), '{}'.format(
                            input_shape),
                        c2_utils.UnscopeName(str(output_name)), '{}'.format(
                            output_shape), suffix))
                    logger.info('{:28s} mean: {}'.format(str(input_name[j]), input_blob.mean()))
                    logger.info('{:28s} max: {}'.format(str(input_name[j]), input_blob.max()))
                    logger.info('{:28s} min: {}'.format(str(input_name[j]), input_blob.min()))
                    logger.info('{:28s} data: {}'.format(str(input_name[j]), input_blob))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info('End of model: {}'.format(model.net.Name()))
