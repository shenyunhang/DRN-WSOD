#!/usr/bin/env python3

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

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine_wsl import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils

from detectron.core.test_engine_wsl import get_inference_dataset
from detectron.core.test_engine_wsl import get_roidb_and_dataset
from detectron.core.test_engine_wsl import empty_results
from detectron.core.test_engine_wsl import extend_results
from detectron.core.test_wsl import box_results_with_nms_and_limit
from detectron.datasets import task_evaluation
from detectron.utils.io import load_object
from detectron.utils.timer import Timer
from collections import defaultdict

import csv

c2_utils.import_detectron_ops()
c2_utils.import_custom_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def grid_search():
    dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, None 
    )
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    subinds = np.array_split(range(num_images), cfg.NUM_GPUS)

    tag = 'detection'
    output_dir = get_output_dir(cfg.TEST.DATASETS, training=False)

    det_file = os.path.join(output_dir, 'detections.pkl')
    outputs = load_object(det_file)

    print(len(outputs))
    all_dets_cache = outputs['all_boxes']
    print(len(all_dets_cache))

    all_boxes_cache = []
    all_scores_cache = []
    for i, entry in enumerate(roidb):
        print(i)
        max_det = all_dets_cache[1][i].shape[0]
        print(max_det, num_classes)
        
        boxes = np.zeros((max_det, 4), dtype=np.float32)
        scores = np.zeros((max_det, num_classes), dtype=np.float32)
        boxes[:] = -1
        scores[:] = -1
        for j in range(num_classes):
            if len(all_dets_cache[j]) > 0:
                pass
            else:
                continue
            scores[:, j] = all_dets_cache[j][i][:, 4]
        boxes[:, 0:4] = all_dets_cache[1][i][:, :4]
        boxes = np.tile(boxes, (1, scores.shape[1]))
        print(scores.shape, boxes.shape)
        all_boxes_cache.append(boxes)
        all_scores_cache.append(scores)

    timers = defaultdict(Timer)
    resultss = []
    nmses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    threshs = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    max_per_images = [10000, 1000, 100, 10, 1]

    for nms in nmses:
        for thresh in threshs:
            for max_per_image in max_per_images:
                print("----------------------------------------------------")
                print('NUM: ', nms, ' Thresh: ', thresh, ' MAX_PER_IM: ', max_per_image)
                cfg.immutable(False)
                cfg.TEST.NMS = nms
                cfg.TEST.SCORE_THRESH = thresh
                cfg.TEST.DETECTIONS_PER_IM = max_per_image
                cfg.immutable(True)

                all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
                for i, entry in enumerate(roidb):
                    # print(i)
                    timers['im_detect_bbox'].tic()
                    scores = all_scores_cache[i]
                    boxes = all_boxes_cache[i]
                    # print(scores.shape, boxes.shape)

                    timers['im_detect_bbox'].toc()

                    timers['misc_bbox'].tic()
                    scores, boxes, cls_boxes_i = box_results_with_nms_and_limit(scores, boxes)
                    timers['misc_bbox'].toc()

                    extend_results(i, all_boxes, cls_boxes_i)

                results = task_evaluation.evaluate_all(
                    dataset, all_boxes, all_segms, all_keyps, output_dir
                )
                print(results)

    print(resultss)
    f = open('grid_search.csv', 'wb')
    wr = csv.writer(f, dialect='excel')
    wr.writerows(resultss)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    # Check for the final model (indicates training already finished)
    final_path = os.path.join(output_dir, 'model_final.pkl')

    while not os.path.exists(final_path) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(final_path))
        time.sleep(10)

    # cfg.immutable(False)
    # cfg.TEST.NMS = 1.1
    # cfg.TEST.SCORE_THRESH = -9999999999.0
    # cfg.TEST.DETECTIONS_PER_IM = 10000000000
    # cfg.immutable(True)
    # run_inference(
        # final_path,
        # ind_range=args.range,
        # multi_gpu_testing=args.multi_gpu_testing,
        # check_expected_results=True,
    # )

    grid_search()
