from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from detectron.core.config import cfg
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.segms as segm_utils

logger = logging.getLogger(__name__)


def add_mask_rcnn_blobs(blobs, sampled_boxes, roidb, im_scale, batch_idx):
    """Add Mask R-CNN specific blobs to the input blob dictionary."""

    # rois_fg = sampled_boxes.copy()

    # Scale rois_fg and format as (batch_idx, x1, y1, x2, y2)
    # rois_fg *= im_scale
    # repeated_batch_idx = batch_idx * blob_utils.ones((rois_fg.shape[0], 1))
    # rois_fg = np.hstack((repeated_batch_idx, rois_fg))

    # Update blobs dict with Mask R-CNN blobs
    # blobs['mask_rois'] = rois_fg
    mask_rois = blobs['rois'].copy()
    blobs['mask_rois'] = mask_rois.astype(np.float32, copy=False)

    # Add label
    labels_oh = blobs['labels_oh']
    mask_labels_oh = np.zeros(
        (mask_rois.shape[0], labels_oh.shape[1]), dtype=np.float32)
    for c in range(labels_oh.shape[1]):
        if labels_oh[0, c] > 0.5:
            mask_labels_oh[:, c] = 1.0
        elif labels_oh[0, c] == 0.5:
            mask_labels_oh[:, c] = 0.5
    blobs['mask_labels_oh'] = mask_labels_oh.astype(np.float32, copy=False)
