from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn_wsl as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_wsl_blob_names(is_training=True):
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if cfg.WSL.CONTEXT and False:
        blob_names += ['rois_frame']
        blob_names += ['rois_context']
    blob_names += ['obn_scores']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
        blob_names += ['labels_oh']
    # if is_training and cfg.MODEL.MASK_ON:
    # 'mask_rois': RoIs sampled for training the mask prediction branch.
    # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
    # blob_names += ['mask_rois']
    # blob_names += ['mask_labels_oh']
    # TODO(YH): NOT SUPPORT
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_wsl_blobs(blobs, im_scales, im_crops, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_crops[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # TODO(YH): NOT SUPPORT
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON:
        valid = keypoint_rcnn_roi_data.finalize_keypoint_minibatch(
            blobs, valid)

    return valid


def _sample_rois(roidb, im_scale, im_crop, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    rois_this_image = np.minimum(rois_per_image, roidb['boxes'].shape[0])

    if False:
        choice = np.random.choice(
            roidb['boxes'].shape[0], rois_this_image, replace=False)
        sampled_boxes = roidb['boxes'][choice, :].copy()
        obn_scores = roidb['obn_scores'][choice, :].copy()
        sampled_scores = np.add(obn_scores, 1.0)
    else:
        sampled_boxes = roidb['boxes'][:rois_this_image].copy()
        obn_scores = roidb['obn_scores'][:rois_this_image].copy()
        sampled_scores = np.add(obn_scores, 1.0)

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    # sampled_rois = sampled_boxes * im_scale
    sampled_rois = _project_im_rois(sampled_boxes, im_scale, im_crop)
    repeated_batch_idx = batch_idx * blob_utils.ones(
        (sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Delete GT Boxes
    # gt_inds = np.where((roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0))[0]
    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    np.delete(sampled_rois, gt_inds, 0)
    np.delete(sampled_scores, gt_inds, 0)

    if cfg.WSL.CONTEXT and False:
        sampled_boxes = roidb['boxes'][:rois_this_image].copy()
        sampled_boxes_inner, sampled_boxes_outer = get_inner_outer_rois(
            sampled_boxes, cfg.WSL.CONTEXT_RATIO)

        sampled_rois_origin = _project_im_rois(sampled_boxes, im_scale,
                                               im_crop)
        sampled_rois_inner = _project_im_rois(sampled_boxes_inner, im_scale,
                                              im_crop)
        sampled_rois_outer = _project_im_rois(sampled_boxes_outer, im_scale,
                                              im_crop)

        repeated_batch_idx_inner = batch_idx * blob_utils.ones(
            (sampled_rois_origin.shape[0], 1))
        repeated_batch_idx_outer = batch_idx * blob_utils.ones(
            (sampled_rois_origin.shape[0], 1))

        sampled_rois_frame = np.hstack(
            (repeated_batch_idx, sampled_rois_origin, sampled_rois_inner))
        sampled_rois_context = np.hstack(
            (repeated_batch_idx, sampled_rois_outer, sampled_rois_origin))

        # Delete GT Boxes
        np.delete(sampled_rois_frame, gt_inds, 0)
        np.delete(sampled_rois_context, gt_inds, 0)

    # Get image label
    img_labels_oh = np.zeros((1, cfg.MODEL.NUM_CLASSES - 1), dtype=np.float32)
    img_labels = np.zeros((1), dtype=np.float32)

    # gt_inds = np.where((roidb['gt_classes'] > 0) & (roidb['is_crowd'] == 0))[0]
    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    assert len(gt_inds) > 0, roidb['gt_classes']
    assert len(
        gt_inds
    ) > 0, 'Empty ground truth empty for image is not allowed. Please check.'
    gt_classes = roidb['gt_classes'][gt_inds].copy()
    num_valid_objs = gt_classes.shape[0]
    for o in range(num_valid_objs):
        img_labels_oh[0][gt_classes[o] - 1] = 1
        img_labels[0] = gt_classes[o] - 1

    blob_dict = dict(
        labels_int32=img_labels.astype(np.int32, copy=False),
        labels_oh=img_labels_oh.astype(np.float32, copy=False),
        rois=sampled_rois.astype(np.float32, copy=False),
        obn_scores=sampled_scores,
    )
    if cfg.WSL.CONTEXT and False:
        blob_dict['rois_frame'] = sampled_rois_frame.astype(
            np.float32, copy=False)
        blob_dict['rois_context'] = sampled_rois_context.astype(
            np.float32, copy=False)

    # Optionally add Mask R-CNN blobs
    # if cfg.MODEL.MASK_ON:
    # mask_rcnn_roi_data.add_mask_rcnn_blobs(blob_dict, sampled_boxes, roidb,
    # im_scale, batch_idx)

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON:
        keypoint_rcnn_roi_data.add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx)

    return blob_dict


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn.map_rois_to_fpn_levels(blobs[rois_blob_name][:, 1:5],
                                                 lvl_min, lvl_max)
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(blobs, rois_blob_name,
                                     blobs[rois_blob_name], target_lvls,
                                     lvl_min, lvl_max)

    _distribute_rois_over_fpn_levels('rois')
    if cfg.MODEL.MASK_ON:
        _distribute_rois_over_fpn_levels('mask_rois')
    if cfg.MODEL.KEYPOINTS_ON:
        _distribute_rois_over_fpn_levels('keypoint_rois')


def _project_im_rois(im_rois, im_scale_factor, im_crop):
    """Project image RoIs into the rescaled training image."""
    im_rois[:, 0] = np.minimum(
        np.maximum(im_rois[:, 0], im_crop[0]), im_crop[2])
    im_rois[:, 1] = np.minimum(
        np.maximum(im_rois[:, 1], im_crop[1]), im_crop[3])
    im_rois[:, 2] = np.maximum(
        np.minimum(im_rois[:, 2], im_crop[2]), im_crop[0])
    im_rois[:, 3] = np.maximum(
        np.minimum(im_rois[:, 3], im_crop[3]), im_crop[1])
    crop = np.tile(im_crop[:2], [im_rois.shape[0], 2])
    rois = (im_rois - crop) * im_scale_factor

    return rois


def get_inner_outer_rois(im_rois, ratio):
    assert ratio > 1, 'ratio should be lager than one in get_inner_outer_rois'
    rois = im_rois.astype(np.float32, copy=True)
    # x1 y1 x2 y2
    rois_w = rois[:, 2] - rois[:, 0]
    rois_h = rois[:, 3] - rois[:, 1]

    rois_inner_w = rois_w / ratio
    rois_inner_h = rois_h / ratio

    rois_outer_w = rois_w * ratio
    rois_outer_h = rois_h * ratio

    inner_residual_w = rois_w - rois_inner_w
    inner_residual_h = rois_h - rois_inner_h

    outer_residual_w = rois_outer_w - rois_w
    outer_residual_h = rois_outer_h - rois_h

    rois_inner = np.copy(rois)
    rois_outer = np.copy(rois)

    # print rois_inner.dtype, rois_inner.shape
    # print inner_residual_w.dtype, inner_residual_w.shape
    # print (inner_residual_w / 2).dtype, (inner_residual_w / 2).shape

    rois_inner[:, 0] += inner_residual_w / 2
    rois_inner[:, 1] += inner_residual_h / 2
    rois_inner[:, 2] -= inner_residual_w / 2
    rois_inner[:, 3] -= inner_residual_h / 2

    rois_outer[:, 0] -= outer_residual_w / 2
    rois_outer[:, 1] -= outer_residual_h / 2
    rois_outer[:, 2] += outer_residual_w / 2
    rois_outer[:, 3] += outer_residual_h / 2

    return rois_inner, rois_outer
