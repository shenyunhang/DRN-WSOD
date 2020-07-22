"""Various network "heads" for predicting masks in Mask R-CNN.

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils

import detectron.modeling.wsl_heads as wsl_head

# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #


def add_seg_outputs(model, blob_in, dim):

    if 'deeplab' in cfg.MRCNN.ROI_MASK_HEAD:
        return add_deeplab_outputs(model, blob_in, dim)
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    if cfg.MRCNN.USE_FC_OUTPUT:
        # Predict masks with a fully connected layer (ignore 'fcn' in the blob
        # name)
        blob_out = model.FC(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls * cfg.MRCNN.RESOLUTION**2,
            weight_init=gauss_fill(0.001),
            bias_init=const_fill(0.0))
    else:
        # Predict mask using Conv

        # Use GaussianFill for class-agnostic mask prediction; fills based on
        # fan-in can be too large in this case and cause divergence
        fill = (cfg.MRCNN.CONV_INIT
                if cfg.MRCNN.CLS_SPECIFIC_MASK else 'GaussianFill')
        blob_out = model.Conv(
            blob_in,
            'mask_fcn_logits',
            dim,
            num_cls - 1,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(fill, {
                'std': 0.001
            }),
            bias_init=const_fill(0.0))

        if cfg.MRCNN.UPSAMPLE_RATIO > 1:
            blob_out = model.BilinearInterpolation(
                'mask_fcn_logits', 'mask_fcn_logits_up', num_cls, num_cls,
                cfg.MRCNN.UPSAMPLE_RATIO)

    if not model.train:  # == if test
        # blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

        # Add BackGround predictions
        model.net.Split(
            blob_out, ['mask_fcn_logits_bg', 'mask_notuse'],
            split=[1, model.num_classes - 2],
            axis=1)
        model.net.Concat(['mask_fcn_logits_bg', blob_out],
                         ['mask_fcn_logits_', 'mask_fcn_logits_concat_dims'],
                         axis=1)

        blob_out = model.net.Sigmoid('mask_fcn_logits_', 'mask_fcn_probs')

    return blob_out


def add_deeplab_outputs(model, blob_in, dim):
    blob_out = model.net.Sum(blob_in, ['mask_fc8'])

    if not model.train:  # == if test
        pass

    if cfg.WSL.MASK_SOFTMAX:
        model.Transpose('mask_fc8', 'mask_fc8_t', axes=(0, 2, 3, 1))
        model.Softmax('mask_fc8_t', 'mask_fc8_probs_t', axis=3)
        model.Transpose(
            'mask_fc8_probs_t', 'mask_fc8_probs', axes=(0, 3, 1, 2))
    else:
        # Add BackGround predictions
        model.net.Sigmoid('mask_fc8', 'mask_fc8_sigmoid')
        model.net.ReduceMax(
            'mask_fc8_sigmoid', 'mask_fc8_fg', axes=[1], keepdims=True)
        model.net.ConstantFill('mask_fc8_fg', 'mask_fc8_one', value=1.0)
        model.net.Sub(['mask_fc8_one', 'mask_fc8_fg'], 'mask_fc8_bg')
        # model.net.Clip('mask_bg', 'mask_bg', min=0.0, max=1.0)

        model.net.Concat(['mask_fc8_bg', 'mask_fc8_sigmoid'],
                         ['mask_fc8_bgfg', 'mask_fc8_bgfg_split_info'],
                         axis=1)

        model.Transpose('mask_fc8_bgfg', 'mask_fc8_bgfg_t', axes=(0, 2, 3, 1))
        model.Softmax('mask_fc8_bgfg_t', 'mask_fc8_probs_t', axis=3)
        model.Transpose(
            'mask_fc8_probs_t', 'mask_fc8_probs', axes=(0, 3, 1, 2))
        # model.net.Copy('mask_fc8_bgfg', 'mask_fc8_probs')

    model.net.Log('mask_fc8_probs', 'mask_fc8_log')
    model.net.Scale('mask_fc8_log', 'mask_fc8_unary', scale=-1.0)
    # model.net.Scale('mask_fc8_bgfg_probs', 'mask_fc8_unary', scale=-1.0)

    model.net.UpsampleBilinearWSL(['data', 'mask_fc8_unary'], 'mask_fc8_data')

    crf_args = {}

    # crf_args['SIZE_STD'] = 513
    # crf_args['POS_W'] = 3
    # crf_args['POS_X_STD'] = 1
    # crf_args['POS_Y_STD'] = 1
    # crf_args['BI_W'] = 4
    # crf_args['BI_X_STD'] = 67
    # crf_args['BI_Y_STD'] = 67
    # crf_args['BI_R_STD'] = 3
    # crf_args['BI_G_STD'] = 3
    # crf_args['BI_B_STD'] = 3

    # crf_args['SIZE_STD'] = 513
    # crf_args['POS_W'] = 2
    # crf_args['POS_X_STD'] = 2
    # crf_args['POS_Y_STD'] = 2
    # crf_args['BI_W'] = 4
    # crf_args['BI_X_STD'] = 65
    # crf_args['BI_Y_STD'] = 65
    # crf_args['BI_R_STD'] = 3
    # crf_args['BI_G_STD'] = 3
    # crf_args['BI_B_STD'] = 3

    # crf_args['SIZE_STD'] = 513
    # crf_args['POS_W'] = 3
    # crf_args['POS_X_STD'] = 3
    # crf_args['POS_Y_STD'] = 3
    # crf_args['BI_W'] = 4
    # crf_args['BI_X_STD'] = 49
    # crf_args['BI_Y_STD'] = 49
    # crf_args['BI_R_STD'] = 5
    # crf_args['BI_G_STD'] = 5
    # crf_args['BI_B_STD'] = 5

    model.net.DenseCRF(['mask_fc8_unary', 'mask_fc8_data'], 'mask_fc8_crf',
                       **crf_args)

    return blob_out


def add_seg_losses(model, blob_mask):
    if 'deeplab' in cfg.MRCNN.ROI_MASK_HEAD:
        return add_deeplab_losses(model, blob_mask)
    """Add Mask R-CNN specific losses."""

    model.MaxPool(blob_mask, 'mask_cls_logits', kernel=28, pad=0, stride=28)
    # model.MaxPool(blob_mask, 'mask_cls_prob', kernel=14, pad=0, stride=14)

    model.net.Squeeze('mask_cls_logits', 'mask_cls_logits_squ', dims=[2, 3])

    # model.net.CrossEntropyWithLogits(['mask_cls_prob', 'rois_pred'],
    # ['mask_cross_entropy'])
    model.net.WeightedSigmoidCrossEntropyWithLogits(
        ['mask_cls_logits_squ', 'mask_labels_oh', 'mask_w'],
        ['mask_cross_entropy'])

    loss_mask = model.net.AveragedLoss(['mask_cross_entropy'],
                                       ['mask_loss_cls'])

    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.AddLosses('mask_loss_cls')
    return loss_gradients


def add_csc_loss(model):
    loss_gradients = {}
    if cfg.WSL.MASK_SOFTMAX:
        model.net.Split(['mask_fc8_up'],
                        ['mask_fc_up_split', 'mask_fc_up_split_notuse'],
                        split=[1, model.num_classes - 1],
                        axis=1)

        model.net.Split(['mask_cross_entropy'],
                        ['mask_cross_entropy_split', 'mask_notuse'],
                        split=[1, model.num_classes - 1],
                        axis=1)

        model.net.CPGSW([
            'cpg', 'mask_fc8_up_split', 'mask_cross_entropy_split',
            'labels_oh', 'cls_prob'
        ],
                        'cpg_sw',
                        max_iter=cfg.WSL.CPG_MAX_ITER,
                        tau=cfg.WSL.CPG_TAU,
                        min_loss=0.1)
    # else:
    # model.net.UpsampleBilinearWSL(['mask_fc8_crf_fg', 'cpg'],
    # 'mask_fc8_crf_fg_up')
    # model.net.CPGSW([
    # 'cpg', 'mask_fc8_up', 'mask_cross_entropy', 'labels_oh', 'cls_prob'
    # ],
    # 'cpg_sw',
    # max_iter=cfg.WSL.CPG_MAX_ITER,
    # tau=cfg.WSL.CPG_TAU,
    # min_loss=999999.0)

    # model.net.CPGScale(['cpg_sw', 'labels_oh', 'cls_prob'],
    # 'cpg_sc',
    # tau=cfg.WSL.CPG_TAU)

    model.net.Sigmoid(['mask_fc8_up'], 'mask_fc8_up_si')

    model.net.CPGScale(['mask_fc8_up_si', 'labels_oh', 'cls_prob'],
                       'mask_fc8_up_sc',
                       tau=cfg.WSL.CPG_TAU)

    loss_gradients_back = wsl_head.add_csc_loss(
        model,
        #'cpg_sc',
        'mask_fc8_up_sc',
        'cls_prob',
        prefix='mask_',
        loss_weight=0.1,
        csc_layer='CSCM',
        area_sqrt=False,
        # area_sqrt=True,
        context_scale=1.8,
        fg_threshold=0.1,
        tau=0.7,
    )
    loss_gradients.update(loss_gradients_back)

    # loss_gradients_csc = wsl_head.add_csc_loss(
    # model, 'cpg', 'cls_prob', prefix='', loss_weight=1.0, area_sqrt=False)
    # loss_gradients.update(loss_gradients_csc)

    return loss_gradients


def add_csc_loss2(model):
    loss_gradients = {}
    if cfg.WSL.MASK_SOFTMAX:
        model.net.Split(['mask_fc8_up'],
                        ['mask_fc_up_split', 'mask_fc_up_split_notuse'],
                        split=[1, model.num_classes - 1],
                        axis=1)
        model.net.CPGScale(['mask_fc8_up_split', 'labels_oh', 'cls_prob'],
                           'mask_fc8_up_scale',
                           tau=cfg.WSL.CPG_TAU)
    else:
        model.net.CPGScale(['mask_fc8_up', 'labels_oh', 'cls_prob'],
                           'mask_fc8_up_scale',
                           tau=cfg.WSL.CPG_TAU)

    loss_gradients_back = wsl_head.add_csc_loss(
        model,
        'mask_fc8_up_scale',
        'cls_prob',
        prefix='mask_',
        loss_weight=0.01)
    loss_gradients.update(loss_gradients_back)

    loss_gradients_csc = wsl_head.add_csc_loss(
        model, 'cpg', 'cls_prob', prefix='')
    loss_gradients.update(loss_gradients_csc)

    return loss_gradients


def add_deeplab_losses(model, blob_mask):
    loss_gradients = {}

    if cfg.WSL.MASK_SOFTMAX:
        model.net.DeeplabUtility(['cpg', 'labels_oh', 'cls_prob'],
                                 ['mask_labels_oh'],
                                 tau=cfg.WSL.CPG_TAU,
                                 softmax=True)

        model.net.UpsampleBilinearWSL(['mask_fc8_probs', 'mask_labels_oh'],
                                      'mask_fc8_up')

        model.net.LabelCrossEntropyWSL(
            ['mask_fc8_up', 'mask_labels_oh', 'cpg'],
            ['mask_cross_entropy', 'entropy_count'])

    else:
        model.net.DeeplabUtility(
            ['cpg', 'labels_oh', 'cls_prob'],
            ['mask_labels_oh'],
            tau=cfg.WSL.CPG_TAU,
        )

        model.net.UpsampleBilinearWSL(['mask_fc8', 'mask_labels_oh'],
                                      'mask_fc8_up')

        model.net.SigmoidCrossEntropyWithLogitsWSL(
            ['mask_fc8_up', 'mask_labels_oh', 'cpg'],
            ['mask_cross_entropy', 'entropy_count'])

    model.net.ReduceSum(
        'mask_cross_entropy',
        'mask_cross_entropy_reduce',
        axes=[1],
        keepdims=True)

    loss_mask_seed = model.net.AveragedLoss(['mask_cross_entropy_reduce'],
                                            ['mask_seed_loss'])

    lg_mask_seed = wsl_head.get_loss_gradients_weighted(
        model, [loss_mask_seed], 1.0)
    # loss_gradients_mask_seed = wsl_head.get_loss_gradients_weighted(model, [loss_mask_seed], 0.1)
    model.AddLosses('mask_seed_loss')

    loss_gradients.update(lg_mask_seed)

    # -------------------------------------------------------------------------------
    # CRF loss

    model.net.Split(['mask_fc8_crf', 'mask_fc8_bgfg_split_info'],
                    ['mask_fc8_crf_bg', 'mask_fc8_crf_fg'],
                    axis=1)

    model.net.KL(['mask_fc8_sigmoid', 'mask_fc8_crf_fg', 'cpg'],
                 ['mask_constraint', 'mask_constraint_count'])
    # model.net.BernoulliJSD(['mask_fc8_sigmoid', 'mask_fc8_crf'],
    # ['mask_constraint'])

    model.net.ReduceMax(
        'mask_fc8_crf', 'mask_crf_sum', axes=[0, 1, 2, 3], keepdims=False)
    model.AddMetrics('mask_crf_sum')
    model.net.ReduceMean(
        'mask_fc8_crf', 'mask_crf_mean', axes=[0, 1, 2, 3], keepdims=False)
    model.AddMetrics('mask_crf_mean')

    model.net.ReduceSum(
        'mask_constraint', 'mask_constraint_reduce', axes=[1], keepdims=True)

    loss_mask_constraint = model.net.AveragedLoss(['mask_constraint_reduce'],
                                                  ['mask_constraint_loss'])

    lg_mask_constraint = wsl_head.get_loss_gradients_weighted(
        model, [loss_mask_constraint], 1.0)
    model.AddLosses('mask_constraint_loss')

    loss_gradients.update(lg_mask_constraint)

    if cfg.WSL.CSC:
        loss_gradients.update(add_csc_loss(model))
        # loss_gradients.update(add_csc_loss2(model))

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Mask heads
# ---------------------------------------------------------------------------- #


def deeplab_vgg16_head(model, blob_in, dim_in, spatial_scale):
    if cfg.WSL.MASK_SOFTMAX:
        num_classes = model.num_classes
    else:
        num_classes = model.num_classes - 1
    if cfg.MRCNN.DILATION == 2 and cfg.WSL.DILATION == 1:
        model.MaxPool('conv4_3', '_[mask]_pool4', kernel=2, pad=0, stride=1)

        model.Conv(
            '_[mask]_pool4',
            '_[mask]_conv5_1',
            512,
            512,
            3,
            pad=2,
            stride=1,
            dilation=2)
        model.Relu('_[mask]_conv5_1', '_[mask]_conv5_1')
        model.Conv(
            '_[mask]_conv5_1',
            '_[mask]_conv5_2',
            512,
            512,
            3,
            pad=2,
            stride=1,
            dilation=2)
        model.Relu('_[mask]_conv5_2', '_[mask]_conv5_2')
        model.Conv(
            '_[mask]_conv5_2',
            '_[mask]_conv5_3',
            512,
            512,
            3,
            pad=2,
            stride=1,
            dilation=2)
        blob_out = model.Relu('_[mask]_conv5_3', '_[mask]_conv5_3')

        model.MaxPool(
            '_[mask]_conv5_3', '_[mask]_pool5', kernel=3, pad=1, stride=1)
    else:
        model.MaxPool('conv5_3', '_[mask]_pool5', kernel=3, pad=1, stride=1)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient('_[mask]_pool5', '_[mask]_pool5')

    # hole = 6
    model.Conv(
        '_[mask]_pool5', 'fc6_1', dim_in, 1024, 3, pad=6, stride=1, dilation=6)
    model.Relu('fc6_1', 'fc6_1')
    l = DropoutIfTraining(model, 'fc6_1', 'drop6_1', 0.5)

    model.Conv(l, 'fc7_1', 1024, 1024, 1, pad=0, stride=1, dilation=1)
    model.Relu('fc7_1', 'fc7_1')
    l = DropoutIfTraining(model, 'fc7_1', 'drop7_1', 0.5)

    model.Conv(
        l, 'mask_fc8_1', 1024, num_classes, 1, pad=0, stride=1, dilation=1)

    # hole = 12
    model.Conv(
        '_[mask]_pool5',
        'fc6_2',
        dim_in,
        1024,
        3,
        pad=12,
        stride=1,
        dilation=12)
    model.Relu('fc6_2', 'fc6_2')
    l = DropoutIfTraining(model, 'fc6_2', 'drop6_2', 0.5)

    model.Conv(l, 'fc7_2', 1024, 1024, 1, pad=0, stride=1, dilation=1)
    model.Relu('fc7_2', 'fc7_2')
    l = DropoutIfTraining(model, 'fc7_2', 'drop7_2', 0.5)

    model.Conv(
        l, 'mask_fc8_2', 1024, num_classes, 1, pad=0, stride=1, dilation=1)

    # hole = 18
    model.Conv(
        '_[mask]_pool5',
        'fc6_3',
        dim_in,
        1024,
        3,
        pad=18,
        stride=1,
        dilation=18)
    model.Relu('fc6_3', 'fc6_3')
    l = DropoutIfTraining(model, 'fc6_3', 'drop6_3', 0.5)

    model.Conv(l, 'fc7_3', 1024, 1024, 1, pad=0, stride=1, dilation=1)
    model.Relu('fc7_3', 'fc7_3')
    l = DropoutIfTraining(model, 'fc7_3', 'drop7_3', 0.5)

    model.Conv(
        l, 'mask_fc8_3', 1024, num_classes, 1, pad=0, stride=1, dilation=1)

    # hole = 24
    model.Conv(
        '_[mask]_pool5',
        'fc6_4',
        dim_in,
        1024,
        3,
        pad=24,
        stride=1,
        dilation=24)
    model.Relu('fc6_4', 'fc6_4')
    l = DropoutIfTraining(model, 'fc6_4', 'drop6_4', 0.5)

    model.Conv(l, 'fc7_4', 1024, 1024, 1, pad=0, stride=1, dilation=1)
    model.Relu('fc7_4', 'fc7_4')
    l = DropoutIfTraining(model, 'fc7_4', 'drop7_4', 0.5)

    model.Conv(
        l, 'mask_fc8_4', 1024, num_classes, 1, pad=0, stride=1, dilation=1)

    blob_mask = ['mask_fc8_1', 'mask_fc8_2', 'mask_fc8_3', 'mask_fc8_4']

    return blob_mask, num_classes


def DropoutIfTraining(model, blob_in, blob_out, dropout_rate):
    """Add dropout to blob_in if the model is in training mode and
    dropout_rate is > 0."""
    if model.train and dropout_rate > 0:
        blob_out = model.Dropout(
            blob_in, blob_out, ratio=dropout_rate, is_test=False)
        return blob_out
    else:
        return blob_in


def mask_rcnn_fcn_head(model, blob_in, dim_in, spatial_scale):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    if model.train:
        model.net.MaskUtility(['rois_pred', 'rois', 'labels_oh'],
                              ['mask_w', 'mask_rois', 'mask_labels_oh'])
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    num_convs = 2
    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {
                'std': 0.001
            }),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = dim_inner

    # return current, dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {
            'std': 0.001
        }),
        bias_init=const_fill(0.0))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v1up4convs(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(model, blob_in, dim_in, spatial_scale,
                                         4)


def mask_rcnn_fcn_head_v1up4convs_gn(model, blob_in, dim_in, spatial_scale):
    """v1up design: 4 * (conv 3x3), convT 2x2, with GroupNorm"""
    return mask_rcnn_fcn_head_v1upXconvs_gn(model, blob_in, dim_in,
                                            spatial_scale, 4)


def mask_rcnn_fcn_head_v1up(model, blob_in, dim_in, spatial_scale):
    """v1up design: 2 * (conv 3x3), convT 2x2."""
    return mask_rcnn_fcn_head_v1upXconvs(model, blob_in, dim_in, spatial_scale,
                                         2)


def mask_rcnn_fcn_head_v1upXconvs(model, blob_in, dim_in, spatial_scale,
                                  num_convs):
    """v1upXconvs design: X * (conv 3x3), convT 2x2."""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.Conv(
            current,
            '_[mask]_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            kernel=3,
            dilation=dilation,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {
                'std': 0.001
            }),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {
            'std': 0.001
        }),
        bias_init=const_fill(0.0))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v1upXconvs_gn(model, blob_in, dim_in, spatial_scale,
                                     num_convs):
    """v1upXconvs design: X * (conv 3x3), convT 2x2, with GroupNorm"""
    current = model.RoIFeatureTransform(
        blob_in,
        blob_out='_mask_roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dilation = cfg.MRCNN.DILATION
    dim_inner = cfg.MRCNN.DIM_REDUCED

    for i in range(num_convs):
        current = model.ConvGN(
            current,
            '_mask_fcn' + str(i + 1),
            dim_in,
            dim_inner,
            group_gn=get_group_gn(dim_inner),
            kernel=3,
            pad=1 * dilation,
            stride=1,
            weight_init=(cfg.MRCNN.CONV_INIT, {
                'std': 0.001
            }),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = dim_inner

    # upsample layer
    model.ConvTranspose(
        current,
        'conv5_mask',
        dim_inner,
        dim_inner,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {
            'std': 0.001
        }),
        bias_init=const_fill(0.0))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner


def mask_rcnn_fcn_head_v0upshare(model, blob_in, dim_in, spatial_scale):
    """Use a ResNet "conv5" / "stage5" head for mask prediction. Weights and
    computation are shared with the conv5 box head. Computation can only be
    shared during training, since inference is cascaded.

    v0upshare design: conv5, convT 2x2.
    """
    # Since box and mask head are shared, these must match
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(['res5_2_sum', 'roi_has_mask_int32'],
                                        ['_[mask]_res5_2_sum_sliced'])
    else:  # re-compute at test time
        blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
            model, blob_in, dim_in, spatial_scale)

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {
            'std': 0.001
        }),  # std only for gauss
        bias_init=const_fill(0.0))
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def mask_rcnn_fcn_head_v0up(model, blob_in, dim_in, spatial_scale):
    """v0up design: conv5, deconv 2x2 (no weight sharing with the box head)."""
    blob_conv5, dim_conv5 = add_ResNet_roi_conv5_head_for_masks(
        model, blob_in, dim_in, spatial_scale)

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5,
        'conv5_mask',
        dim_conv5,
        dim_reduced,
        kernel=2,
        pad=0,
        stride=2,
        weight_init=('GaussianFill', {
            'std': 0.001
        }),
        bias_init=const_fill(0.0))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced


def add_ResNet_roi_conv5_head_for_masks(model, blob_in, dim_in, spatial_scale):
    """Add a ResNet "conv5" / "stage5" head for predicting masks."""
    model.RoIFeatureTransform(
        blob_in,
        blob_out='_[mask]_pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dilation = cfg.MRCNN.DILATION
    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2

    s, dim_in = ResNet.add_stage(
        model,
        '_[mask]_res5',
        '_[mask]_pool5',
        3,
        dim_in,
        2048,
        512,
        dilation,
        stride_init=stride_init)

    return s, 2048
