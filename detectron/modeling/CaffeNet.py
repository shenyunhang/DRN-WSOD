from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg


def add_CaffeNet_conv5_body(model):
    # weight_init = ('XavierFill', {})
    weight_init = ('GaussianFill', {'std': 0.01})

    bias_init = ('ConstantFill', {'value': 0.0})
    conv1 = model.Conv('data', 'conv1', 3, 96, 11, pad=0, stride=4, weight_init=weight_init, bias_init=bias_init)
    relu1 = model.Relu(conv1, 'conv1')
    pool1 = model.MaxPool(relu1, 'pool1', kernel=3, pad=0, stride=2)
    norm1 = model.LRN(pool1, 'norm1', size=5, alpha=0.0001, beta=0.75)

    bias_init = ('ConstantFill', {'value': 1.0})
    conv2 = model.Conv(pool1, 'conv2', 96, 256, kernel=5, group=2, pad=2, stride=1, weight_init=weight_init, bias_init=bias_init)
    relu2 = model.Relu(conv2, 'conv2')
    pool2 = model.MaxPool(relu2, 'pool2', kernel=3, pad=0, stride=2)
    norm2 = model.LRN(pool2, 'norm2', size=5, alpha=0.0001, beta=0.75)

    bias_init = ('ConstantFill', {'value': 0.0})
    conv3 = model.Conv(pool2, 'conv3', 256, 384, kernel=3, pad=1, stride=1, weight_init=weight_init, bias_init=bias_init)
    relu3 = model.Relu(conv3, 'conv3')

    bias_init = ('ConstantFill', {'value': 1.0})
    conv4 = model.Conv(relu3, 'conv4', 384, 384, kernel=3, group=2, pad=1, stride=1, weight_init=weight_init, bias_init=bias_init)
    relu4 = model.Relu(conv4, 'conv4')

    bias_init = ('ConstantFill', {'value': 1.0})
    conv5 = model.Conv(relu4, 'conv5', 384, 256, kernel=3, group=2, pad=1, stride=1, weight_init=weight_init, bias_init=bias_init)
    relu5 = model.Relu(conv5, 'conv5')
    return relu5, 256, 1. / 16.


def add_CaffeNet_fc_head(model, blob_in, dim_in, spatial_scale):
    pool5 = model.MaxPool(blob_in, 'pool5', kernel=3, pad=0, stride=2)

    # weight_init = ('XavierFill', {})
    weight_init = ('GaussianFill', {'std': 0.005})

    bias_init = ('ConstantFill', {'value': 1.0})
    fc6 = model.FC(pool5, 'fc6', dim_in * 6 * 6, 4096, weight_init=weight_init, bias_init=bias_init)
    relu6 = model.Relu(fc6, 'fc6')
    drop6 = model.Dropout(relu6, 'drop6', ratio=0.5, is_test=not model.train)

    bias_init = ('ConstantFill', {'value': 1.0})
    fc7 = model.FC(drop6, 'fc7', 4096, 4096, weight_init=weight_init, bias_init=bias_init)
    relu7 = model.Relu(fc7, 'fc7')
    drop7 = model.Dropout(relu7, 'drop7', ratio=0.5, is_test=not model.train)
    return drop7, 4096, 1. / 32.


def add_CaffeNet_roi_fc_head(model, blob_in, dim_in, spatial_scale):
    pool5 = model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=6,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    fc6 = model.FC(pool5, 'fc6', dim_in * 6 * 6, 4096)
    relu6 = model.Relu(fc6, 'fc6')
    drop6 = model.Dropout(relu6, 'drop6', ratio=0.5, is_test=not model.train)

    fc7 = model.FC(drop6, 'fc7', 4096, 4096)
    relu7 = model.Relu(fc7, 'fc7')
    drop7 = model.Dropout(relu7, 'drop7', ratio=0.5, is_test=not model.train)

    return drop7, 4096
