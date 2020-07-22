from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging

import detectron.utils.boxes as box_utils
import detectron.roi_data.data_utils as data_utils
from detectron.core.config import cfg

logger = logging.getLogger(__name__)


def get_wsl_blob_names(is_training=True):
    """
    Returns blob names in the order in which they are read by the data
    loader.
    """
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    # Same format as RPN blobs, but one per FPN level
    if is_training:
        blob_names += ['cls_labels']
    return blob_names


def add_wsl_blobs(blobs, im_scales, roidb, image_width, image_height):
    """Add WSL blobs."""
    for im_i, entry in enumerate(roidb):
        cls_labels = np.zeros((1, cfg.MODEL.NUM_CLASSES - 1), dtype=np.float32)
        gt_inds = np.where((entry['gt_classes'] > 0) &
                           (entry['is_crowd'] == 0))[0]
        assert len(
            gt_inds
        ) > 0, 'Empty ground truth empty for image is not allowed. Please check.'
        gt_classes = entry['gt_classes'][gt_inds]
        num_valid_objs = gt_classes.shape[0]
        for o in range(num_valid_objs):
            cls_labels[0][gt_classes[o] - 1] = 1
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)

        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)
        blobs['cls_labels'].append(cls_labels)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v, axis=0)
            # logger.info(k)
            # logger.info(v)

    return True
