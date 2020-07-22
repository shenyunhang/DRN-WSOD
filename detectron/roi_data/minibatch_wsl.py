"""Construct minibatches for Detectron networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import os
import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.roi_data.retinanet as retinanet_roi_data
import detectron.roi_data.rpn as rpn_roi_data
import detectron.roi_data.retinanet_wsl as retinanet_wsl_roi_data
import detectron.roi_data.wsl as wsl_roi_data
import detectron.utils.blob as blob_utils

logger = logging.getLogger(__name__)


def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    blob_names += ['data_ids']
    if cfg.WSL.WSL_ON:
        if cfg.RETINANET.RETINANET_ON:
            blob_names += retinanet_wsl_roi_data.get_wsl_blob_names(
                is_training=is_training)
        else:
            blob_names += wsl_roi_data.get_wsl_blob_names(
                is_training=is_training)
    elif cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += rpn_roi_data.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        blob_names += retinanet_roi_data.get_retinanet_blob_names(
            is_training=is_training
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += fast_rcnn_roi_data.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    # im_crops is define as RoIs with form (y1,x1,y2,x2)
    im_blob, im_scales, im_crops = _get_image_blob(roidb)

    # row col row col to x1 y1 x2 y2
    im_crops = np.array(im_crops, dtype=np.int32)
    im_crops = im_crops[:, (1, 0, 3, 2)]

    blobs['data'] = im_blob
    blobs['data_ids'] = _get_image_id_blob(roidb)
    if cfg.WSL.WSL_ON:
        if cfg.RETINANET.RETINANET_ON:
            im_width, im_height = im_blob.shape[3], im_blob.shape[2]
            valid = retinanet_wsl_roi_data.add_wsl_blobs(
                blobs, im_scales, roidb, im_width, im_height)
        else:
            valid = wsl_roi_data.add_wsl_blobs(blobs, im_scales, im_crops,
                                               roidb)
    elif cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = rpn_roi_data.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        im_width, im_height = im_blob.shape[3], im_blob.shape[2]
        # im_width, im_height corresponds to the network input: padded image
        # (if needed) width and height. We pass it as input and slice the data
        # accordingly so that we don't need to use SampleAsOp
        valid = retinanet_roi_data.add_retinanet_blobs(
            blobs, im_scales, roidb, im_width, im_height
        )
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_id_blob(roidb):
    num_images = len(roidb)
    blob = np.zeros((0, 1), dtype=np.int32)
    for i in range(num_images):
        image_path = roidb[i]['image']
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]
        image_id = int(image_id.split('_')[-1])
        image_id = np.array([image_id], dtype=np.int32)

        blob = np.vstack((blob, image_id))

    return blob


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images
    )
    processed_ims = []
    im_scales = []
    im_crops = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        if cfg.WSL.USE_DISTORTION:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            s0 = npr.random() * (cfg.WSL.SATURATION - 1) + 1
            s1 = npr.random() * (cfg.WSL.EXPOSURE - 1) + 1
            s0 = s0 if npr.random() > 0.5 else 1.0 / s0
            s1 = s1 if npr.random() > 0.5 else 1.0 / s1
            hsv = np.array(hsv, dtype=np.float32)
            hsv[:, :, 1] = np.minimum(s0 * hsv[:, :, 1], 255)
            hsv[:, :, 2] = np.minimum(s1 * hsv[:, :, 2], 255)
            hsv = np.array(hsv, dtype=np.uint8)
            im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if cfg.WSL.USE_CROP:
            im_shape = np.array(im.shape)
            crop_dims = im_shape[:2] * cfg.WSL.CROP

            r0 = npr.random()
            r1 = npr.random()
            s = im_shape[:2] - crop_dims
            s[0] *= r0
            s[1] *= r1
            im_crop = np.array(
                [s[0], s[1], s[0] + crop_dims[0] - 1, s[1] + crop_dims[1] - 1],
                dtype=np.int32)

            im = im[im_crop[0]:im_crop[2] + 1, im_crop[1]:im_crop[3] + 1, :]
        else:
            im_crop = np.array([0, 0, im.shape[0] - 1, im.shape[1] - 1],
                               dtype=np.int32)

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        )
        im_scales.append(im_scale)
        im_crops.append(im_crop)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales, im_crops
