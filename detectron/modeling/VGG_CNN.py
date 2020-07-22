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
"""VGG_CNN from https://arxiv.org/abs/1405.3531."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_VGG_CNN_S_conv5_body(model):
    model.Conv('data', 'conv1', 3, 96, 7, pad=0, stride=2)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=0, stride=3)
    model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 96, 256, 5, pad=0, stride=1)
    model.Relu('conv2', 'conv2')
    model.MaxPool('conv2', 'pool2', kernel=2, pad=0, stride=2)
    model.Conv('pool2', 'conv3', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv3', 'conv3')
    model.Conv('conv3', 'conv4', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4', 'conv4')
    model.Conv('conv4', 'conv5', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5', 'conv5')
    return blob_out, 512, 1. / 12.


def add_VGG_CNN_S_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=3)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return drop7, 4096, 1. / 32.


def add_VGG_CNN_S_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096


def add_VGG_CNN_M_conv5_body(model):
    model.Conv('data', 'conv1', 3, 96, 7, pad=0, stride=2)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=0, stride=2)
    if cfg.TRAIN.FREEZE_AT == 1:
        model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 96, 256, 5, pad=0, stride=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0005, beta=0.75, bias=2.)

    if cfg.WSL.DILATION == 2:
        model.Conv('norm2', 'conv3', 256, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv3', 'conv3')
        model.Conv('conv3', 'conv4', 512, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv4', 'conv4')
        model.Conv('conv4', 'conv5', 512, 512, 3, pad=2, stride=1, dilation=2)
        blob_out = model.Relu('conv5', 'conv5')

        return blob_out, 512, 1. / 8.
    else:
        model.MaxPool('norm2', 'pool2', kernel=3, pad=0, stride=2)
        model.Conv('pool2', 'conv3', 256, 512, 3, pad=1, stride=1)
        model.Relu('conv3', 'conv3')
        model.Conv('conv3', 'conv4', 512, 512, 3, pad=1, stride=1)
        model.Relu('conv4', 'conv4')
        model.Conv('conv4', 'conv5', 512, 512, 3, pad=1, stride=1)
        blob_out = model.Relu('conv5', 'conv5')

        return blob_out, 512, 1. / 16.


def add_VGG_CNN_M_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096, 1. / 32.


def add_VGG_CNN_M_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096


def add_VGG_CNN_M_2048_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 2048)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 2048, 1. / 32.


def add_VGG_CNN_M_2048_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 2048)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 2048


def add_VGG_CNN_M_1024_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 1024)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 1024, 1. / 32.


def add_VGG_CNN_M_1024_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 1024)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 1024


def add_VGG_CNN_M_128_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 128)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 128, 1. / 32.


def add_VGG_CNN_M_128_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 128)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 128


def add_VGG_CNN_F_conv5_body(model):
    model.Conv('data', 'conv1', 3, 64, 11, pad=0, stride=4)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=0, stride=2)
    model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 64, 256, 5, pad=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm2', 'pool2', kernel=3, pad=0, stride=2)
    model.Conv('pool2', 'conv3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3', 'conv3')
    model.Conv('conv3', 'conv4', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv4', 'conv4')
    model.Conv('conv4', 'conv5', 256, 256, 3, pad=1, stride=1)
    blob_out = model.Relu('conv5', 'conv5')
    return blob_out, 256, 1. / 16.


def add_VGG_CNN_F_conv4_body(model):
    model.Conv('data', 'conv1', 3, 64, 11, pad=0, stride=4)
    model.Relu('conv1', 'conv1')
    model.LRN('conv1', 'norm1', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm1', 'pool1', kernel=3, pad=0, stride=2)
    model.StopGradient('pool1', 'pool1')
    # No updates at conv1 and below (norm1 and pool1 have no params,
    # so we can stop gradients before them, too)
    model.Conv('pool1', 'conv2', 64, 256, 5, pad=2)
    model.Relu('conv2', 'conv2')
    model.LRN('conv2', 'norm2', size=5, alpha=0.0005, beta=0.75, bias=2.)
    model.MaxPool('norm2', 'pool2', kernel=3, pad=0, stride=2)
    model.Conv('pool2', 'conv3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3', 'conv3')
    model.Conv('conv3', 'conv4', 256, 256, 3, pad=1, stride=1)
    blob_out = model.Relu('conv4', 'conv4')
    return blob_out, 256, 1. / 16.


def add_VGG_CNN_F_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096, 1. / 32.


def add_VGG_CNN_F_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.FC('pool5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 1024)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096


def add_VGG_CNN_F_roi_conv5_2fc_head(model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in,
        'pool4',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    model.Conv('pool4', 'conv5', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv5', 'conv5')

    model.net.RoIFeatureBoost(['conv5', 'obn_scores'], 'conv5')

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient('conv5', 'conv5')

    model.FC('conv5', 'fc6', dim_in * 6 * 6, 4096)
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'drop6', ratio=0.5, is_test=not model.train)

    model.FC('drop6', 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'drop7', ratio=0.5, is_test=not model.train)

    return 'drop7', 4096
