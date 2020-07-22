from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_VGG16_conv5_body_origin(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    model.Relu('conv1_1', 'conv1_1')
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=2, pad=0, stride=2)

    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=2, pad=0, stride=2)

    if cfg.TRAIN.FREEZE_AT == 2:
        model.StopGradient('pool2', 'pool2')

    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=2, pad=0, stride=2)

    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_3', 'conv4_3')
    if cfg.WSL.DILATION == 2:
        model.MaxPool('conv4_3', 'pool4', kernel=2, pad=0, stride=1)

        model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv5_1', 'conv5_1')
        model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv5_2', 'conv5_2')
        model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=2, stride=1, dilation=2)
        blob_out = model.Relu('conv5_3', 'conv5_3')
        return blob_out, 512, 1. / 8.
    else:
        model.MaxPool('conv4_3', 'pool4', kernel=2, pad=0, stride=2)

        model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)
        model.Relu('conv5_1', 'conv5_1')
        model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)
        model.Relu('conv5_2', 'conv5_2')
        model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)
        blob_out = model.Relu('conv5_3', 'conv5_3')
        return blob_out, 512, 1. / 16.


def add_VGG16_conv4_body_origin(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    model.Relu('conv1_1', 'conv1_1')
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=2, pad=0, stride=2)

    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=2, pad=0, stride=2)

    if cfg.TRAIN.FREEZE_AT == 2:
        model.StopGradient('pool2', 'pool2')

    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=2, pad=0, stride=2)

    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    blob_out = model.Relu('conv4_3', 'conv4_3')
    return blob_out, 512, 1. / 8.


def add_VGG16_conv5_body_deeplab(model):
    model.Conv('data', 'conv1_1', 3, 64, 3, pad=1, stride=1)
    model.Relu('conv1_1', 'conv1_1')
    model.Conv('conv1_1', 'conv1_2', 64, 64, 3, pad=1, stride=1)
    model.Relu('conv1_2', 'conv1_2')
    model.MaxPool('conv1_2', 'pool1', kernel=3, pad=1, stride=2)

    model.Conv('pool1', 'conv2_1', 64, 128, 3, pad=1, stride=1)
    model.Relu('conv2_1', 'conv2_1')
    model.Conv('conv2_1', 'conv2_2', 128, 128, 3, pad=1, stride=1)
    model.Relu('conv2_2', 'conv2_2')
    model.MaxPool('conv2_2', 'pool2', kernel=3, pad=1, stride=2)

    model.Conv('pool2', 'conv3_1', 128, 256, 3, pad=1, stride=1)
    model.Relu('conv3_1', 'conv3_1')
    model.Conv('conv3_1', 'conv3_2', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_2', 'conv3_2')
    model.Conv('conv3_2', 'conv3_3', 256, 256, 3, pad=1, stride=1)
    model.Relu('conv3_3', 'conv3_3')
    model.MaxPool('conv3_3', 'pool3', kernel=3, pad=1, stride=2)

    model.Conv('pool3', 'conv4_1', 256, 512, 3, pad=1, stride=1)
    model.Relu('conv4_1', 'conv4_1')
    model.Conv('conv4_1', 'conv4_2', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_2', 'conv4_2')
    model.Conv('conv4_2', 'conv4_3', 512, 512, 3, pad=1, stride=1)
    model.Relu('conv4_3', 'conv4_3')
    if cfg.WSL.DILATION == 2:
        model.MaxPool('conv4_3', 'pool4', kernel=3, pad=1, stride=1)

        model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv5_1', 'conv5_1')
        model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=2, stride=1, dilation=2)
        model.Relu('conv5_2', 'conv5_2')
        model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=2, stride=1, dilation=2)
        blob_out = model.Relu('conv5_3', 'conv5_3')
        return blob_out, 512, 1. / 8.
    else:
        model.MaxPool('conv4_3', 'pool4', kernel=3, pad=1, stride=2)

        model.Conv('pool4', 'conv5_1', 512, 512, 3, pad=1, stride=1)
        model.Relu('conv5_1', 'conv5_1')
        model.Conv('conv5_1', 'conv5_2', 512, 512, 3, pad=1, stride=1)
        model.Relu('conv5_2', 'conv5_2')
        model.Conv('conv5_2', 'conv5_3', 512, 512, 3, pad=1, stride=1)
        blob_out = model.Relu('conv5_3', 'conv5_3')
        return blob_out, 512, 1. / 16.


def add_VGG16_roi_2fc_head(model, blob_in, dim_in, spatial_scale):
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    model.FC('pool5', 'fc6', dim_in * 7 * 7, 4096)
    model.Relu('fc6', 'fc6')
    l = DropoutIfTraining(model, 'fc6', 'drop6', 0.5)
    model.FC(l, 'fc7', 4096, 4096)
    model.Relu('fc7', 'fc7')
    blob_out = DropoutIfTraining(model, 'fc7', 'drop7', 0.5)
    return blob_out, 4096


def add_VGG16_roi_conv5_2fc_head(model, blob_in, dim_in, spatial_scale):
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        'pool4',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.Conv(l, 'conv5_1', 512, 512, 3, pad=1, stride=1)
    l = model.Relu(l, 'conv5_1')
    l = model.Conv(l, 'conv5_2', 512, 512, 3, pad=1, stride=1)
    l = model.Relu(l, 'conv5_2')
    l = model.Conv(l, 'conv5_3', 512, 512, 3, pad=1, stride=1)
    l = model.Relu(l, 'conv5_3')
    l = model.MaxPool(l, 'pool5', kernel=2, pad=0, stride=2)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.FC(l, 'fc6', dim_in * 7 * 7, 4096)
    l = model.Relu(l, 'fc6')
    l = DropoutIfTraining(model, l, 'drop6', 0.5)
    l = model.FC(l, 'fc7', 4096, 4096)
    l = model.Relu(l, 'fc7')
    l = DropoutIfTraining(model, 'fc7', 'drop7', 0.5)
    return l, 4096


def DropoutIfTraining(model, blob_in, blob_out, dropout_rate):
    """Add dropout to blob_in if the model is in training mode and
    dropout_rate is > 0."""
    if model.train and dropout_rate > 0:
        blob_out = model.Dropout(
            blob_in, blob_out, ratio=dropout_rate, is_test=False)
        return blob_out
    else:
        return blob_in
