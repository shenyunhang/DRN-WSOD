from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import math

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.utils.io import save_object


def feat_map_draw(im_name, output_dir, feat_map, is_save=True):
    feat_map = np.max(feat_map.copy(), 0)

    max_value = np.max(feat_map)
    if max_value > 0:
        # max_value = max_value * 0.1
        # feat_map = np.clip(feat_map, 0, max_value)
        feat_map = feat_map / max_value * 255
    feat_map = feat_map.astype(np.uint8)
    im_color = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)
    if not is_save:
        return im_color
    file_name = os.path.join(output_dir, im_name + '.png')
    cv2.imwrite(file_name, im_color)


def argmax_feat_map_draw(im_name, output_dir, im, conv5, roi_feat,
                         argmax_roi_feat):
    conv5 = feat_map_draw(None, None, conv5.copy(), is_save=False)

    max_idx = np.argmax(roi_feat, axis=0)

    h, w, _ = conv5.shape
    c, ph, pw = argmax_roi_feat.shape
    ih, iw, _ = im.shape

    stride = 1.0 * min(ih, iw) / min(h, w)
    r = 255
    g = 0
    b = 0

    thickness = 4
    # thickness = 1

    for i in range(ph):
        for j in range(pw):
            c = max_idx[i, j]
            idx = argmax_roi_feat[c, i, j]
            idx_h = int(idx / w)
            idx_w = int(idx % w)
            conv5[idx_h, idx_w, 0] = b
            conv5[idx_h, idx_w, 1] = g
            conv5[idx_h, idx_w, 2] = r

            idx_h = max(0, min(int(idx_h * stride), ih - 1))
            idx_w = max(0, min(int(idx_w * stride), iw - 1))
            # im[idx_h, idx_w, 0] = b
            # im[idx_h, idx_w, 1] = g
            # im[idx_h, idx_w, 2] = r

            x1 = int(idx_w - stride / 2 * thickness)
            y1 = int(idx_h - stride / 2 * thickness)
            x2 = int(idx_w + stride / 2 * thickness)
            y2 = int(idx_h + stride / 2 * thickness)

            cv2.rectangle(im, (x1, y1), (x2, y2), (b, g, r), cv2.FILLED)

    file_name = os.path.join(output_dir, im_name + '_im.png')
    cv2.imwrite(file_name, im)

    file_name = os.path.join(output_dir, im_name + '_conv5.png')
    cv2.imwrite(file_name, conv5)


def feat_draw(im_name, output_dir, im, conv5, roi_feats, rois,
              argmax_roi_feats):

    for i in range(roi_feats.shape[0]):
        roi_feat = roi_feats[i].copy()
        feat_map_draw(im_name + '_' + str(i), output_dir, roi_feat)

        argmax_roi_feat = argmax_roi_feats[i].copy()
        argmax_feat_map_draw(im_name + '_argmax_' + str(i), output_dir,
                             im.copy(), conv5.copy(), roi_feat.copy(),
                             argmax_roi_feat.copy())

    feat_map_draw(im_name + '_conv5', output_dir, conv5.copy())


def feat_vis(im, entry, im_name, output_dir):

    if cfg.TEST.BBOX_AUG.ENABLED:
        print('test aug is not support')
        exit(0)

    if cfg.DEDUP_BOXES > 0:
        print('cfg.DEDUP_BOXES > 0 is not support')
        exit(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rois = workspace.FetchBlob(core.ScopedName('rois')).squeeze()
    obn_scores = workspace.FetchBlob(core.ScopedName('obn_scores')).squeeze()

    # conv5 = workspace.FetchBlob(core.ScopedName('res5_1_sum')).squeeze()
    # conv5 = workspace.FetchBlob(core.ScopedName('res5_2_sum')).squeeze()
    # conv5 = workspace.FetchBlob(core.ScopedName('conv5_3')).squeeze()

    roi_feats = workspace.FetchBlob(core.ScopedName('roi_feat')).squeeze()
    fc6 = workspace.FetchBlob(core.ScopedName('fc6')).squeeze()
    fc7 = workspace.FetchBlob(core.ScopedName('fc7')).squeeze()

    argmax_roi_feats = workspace.FetchBlob(
        core.ScopedName('_argmax_roi_feat')).squeeze()

    # Softmax class probabilities
    scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])

    # Select foreground RoIs as those with >= FG_THRESH overlap
    gt_inds = np.where(entry['gt_classes'] > 0)[0]

    max_overlaps = entry['max_overlaps']

    if np.random.random() < 0.10 or True:
        bg_inds = np.where(max_overlaps < 0.5)[0]

        # random bg
        # bg_inds = np.random.choice(bg_inds, 1)

        # hard bg
        scores_bg = scores[bg_inds, :]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_classes = np.unique(gt_classes)
        scores_bg = scores_bg[:, gt_classes]
        # scores_bg = np.sum(scores_bg, axis=1)
        bg_inds = bg_inds[scores_bg.argmax(axis=0)]

        keep_inds = np.append(gt_inds, bg_inds)
    else:
        keep_inds = gt_inds

    rois = rois[keep_inds, ...]
    obn_scores = obn_scores[keep_inds, ...]
    roi_feats = roi_feats[keep_inds, ...]
    argmax_roi_feats = argmax_roi_feats[keep_inds, ...]
    fc6 = fc6[keep_inds, ...]
    fc7 = fc7[keep_inds, ...]
    scores = scores[keep_inds, :]

    # max_classes = entry['max_classes'][keep_inds]
    gt_classes = entry['gt_classes'][keep_inds]

    # feat_draw(im_name, output_dir, im, conv5, roi_feats, rois,
              # argmax_roi_feats)

    save_file = os.path.join(output_dir, im_name + '.pkl')
    print('save to ', save_file)
    save_object(
        dict(
            rois=rois,
            # conv5=conv5,
            roi_feats=roi_feats,
            fc6=fc6,
            fc7=fc7,
            gt_classes=gt_classes,
        ), save_file)
