from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (DenseNet50, DenseNet101, ...)
# ---------------------------------------------------------------------------- #


def add_DenseNet121_conv4_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 24), 64, 128, 32)


def add_DenseNet121_conv5_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 24, 16), 64, 128, 32)


def add_DenseNet161_conv4_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 36), 96, 192, 48)


def add_DenseNet161_conv5_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 36, 24), 96, 192, 48)


def add_DenseNet169_conv4_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 32), 64, 128, 32)


def add_DenseNet169_conv5_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 32, 32), 64, 128, 32)


def add_DenseNet201_conv4_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 48), 64, 128, 32)


def add_DenseNet201_conv5_body(model):
    return add_DenseNet_convX_body(model, (6, 12, 48, 32), 64, 128, 32)


# ---------------------------------------------------------------------------- #
# Generic DenseNet components
# ---------------------------------------------------------------------------- #


def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_group,
    dim_inner,
    dilation,
    stride_init=1,
    has_pool=False,
    stride_pool=2
):
    """Add a DenseNet stage to the model by stacking n residual blocks."""

    if has_pool:
        blob_in = model.MaxPool(blob_in, prefix + '_pool', kernel=2, pad=0, stride=stride_pool)

    for i in range(1, n + 1):
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            blob_in,
            dim_in,
            dim_group,
            dim_inner,
            dilation,
            stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_in + dim_group

    cur = model.AffineChannel(blob_in, prefix + '_blk_bn', dim=dim_in, inplace=True)
    cur = model.Relu(cur, cur)

    if 'conv5' in prefix:
        return cur, dim_in

    dim_out = int(dim_in / 2)
    cur = model.Conv(
        cur,
        prefix + '_blk',
        dim_in,
        dim_out,
        kernel=1,
        stride=stride_init,
        pad=0,
        no_bias=1,
    )


    return cur, dim_out


def add_DenseNet_convX_body(model, block_counts, dim_in, dim_inner, group_rate):
    """Add a DenseNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    freeze_at = cfg.TRAIN.FREEZE_AT
    assert freeze_at in [0, 2, 3, 4, 5]

    # add the stem (by default, conv1 and pool1 with bn; can support gn)
    p, dim_in = globals()[cfg.RESNETS.STEM_FUNC](model, 'data', dim_in)

    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'conv2', p, n1, dim_in, group_rate, dim_inner, 1, has_pool=True)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'conv3', s, n2, dim_in, group_rate, dim_inner, 1, has_pool=True
    )
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'conv4', s, n3, dim_in, group_rate, dim_inner, cfg.RESNETS.RES5_DILATION, has_pool=True, stride_pool=2 if cfg.RESNETS.RES5_DILATION == 1 else 1
    )
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'conv5', s, n4, dim_in, group_rate, dim_inner,
            cfg.RESNETS.RES5_DILATION
        )
        if freeze_at == 5:
            model.StopGradient(s, s)
        return s, dim_in, 1. / 16. * cfg.RESNETS.RES5_DILATION
    else:
        return s, dim_in, 1. / 8.


def add_DenseNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        model, 'res5', 'pool5', 3, dim_in, 2048, dim_bottleneck * 8, 1,
        stride_init
    )
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048


def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2,
    inplace_sum=False
):
    """Add a residual block to the model."""

    stride=1

    # transformation blob
    tr = globals()[cfg.RESNETS.TRANS_FUNC](
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        group=cfg.RESNETS.NUM_GROUPS,
        dilation=dilation
    )

    sc = model.Concat([blob_in, tr], 'concate_' + prefix[4:], axis=1)

    return sc


# ------------------------------------------------------------------------------
# various shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def basic_bn_stem(model, data, dim_in, **kwargs):
    """Add a basic DenseNet stem. For a pre-trained network that used BN.
    An AffineChannel op replaces BN during fine-tuning.
    """

    # weight_init = None
    # weight_init = ('XavierFill', {})
    weight_init = ("MSRAFill", {})

    dim = 64
    p = model.Conv('data', 'conv1_1', 3, dim, 3, pad=1, stride=2, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'conv1_1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)

    p = model.Conv( p, 'conv1_2', dim, dim, 3, pad=1, stride=1, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'conv1_2_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)

    p = model.Conv( p, 'conv1_3', dim, dim, 3, pad=1, stride=1, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'conv1_3_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    # p = model.MaxPool(p, 'pool1_3', kernel=2, pad=0, stride=2)
    return p, dim

    dim = dim_in
    p = model.Conv(data, 'conv1', 3, dim, 7, pad=3, stride=2, no_bias=1)
    p = model.AffineChannel(p, 'conv1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    # p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


def basic_gn_stem(model, data, **kwargs):
    """Add a basic DenseNet stem (using GN)"""

    dim = 64
    p = model.ConvGN(
        data, 'conv1', 3, dim, 7, group_gn=get_group_gn(dim), pad=3, stride=2
    )
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def bottleneck_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation to the model."""


    cur = model.AffineChannel(blob_in, prefix + '_x1_bn', dim=dim_in, inplace=False)
    cur = model.Relu(cur, cur)

    # conv 1x1 -> BN -> ReLU
    cur = model.Conv(
        cur,
        prefix + '_x1',
        dim_in,
        dim_inner,
        kernel=1,
        stride=stride,
        pad=0,
        no_bias=1,
    )
    cur = model.AffineChannel(cur, prefix + '_x2_bn', dim=dim_inner, inplace=True)
    cur = model.Relu(cur, cur)

    cur = model.Conv(
        cur,
        prefix + '_x2',
        dim_inner,
        dim_out,
        kernel=3,
        stride=stride,
        pad=1,
        no_bias=1,
    )

    return cur


def bottleneck_gn_transformation(
    model,
    blob_in,
    dim_in,
    dim_out,
    stride,
    prefix,
    dim_inner,
    dilation=1,
    group=1
):
    """Add a bottleneck transformation with GroupNorm to the model."""
    # In original resnet, stride=2 is on 1x1.
    # In fb.torch resnet, stride=2 is on 3x3.
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> GN -> ReLU
    cur = model.ConvGN(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_inner,
        kernel=1,
        group_gn=get_group_gn(dim_inner),
        stride=str1x1,
        pad=0,
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> GN -> ReLU
    cur = model.ConvGN(
        cur,
        prefix + '_branch2b',
        dim_inner,
        dim_inner,
        kernel=3,
        group_gn=get_group_gn(dim_inner),
        stride=str3x3,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
    )
    cur = model.Relu(cur, cur)

    # conv 1x1 -> GN (no ReLU)
    cur = model.ConvGN(
        cur,
        prefix + '_branch2c',
        dim_inner,
        dim_out,
        kernel=1,
        group_gn=get_group_gn(dim_out),
        stride=1,
        pad=0,
    )
    return cur


def add_Densenet_2fc_head(model, blob_in, dim_in, spatial_scale, dim_fc6=4096, dim_fc7=4096):
    pool5 = model.MaxPool(blob_in, 'pool5', kernel=2, pad=0, stride=2)

    # weight_init = None
    weight_init = ('GaussianFill', {'std': 0.005})
    # weight_init = ('XavierFill', {})
    # weight_init = ("MSRAFill", {})

    bias_init = ('ConstantFill', {'value': 0.1})

    fc6 = model.FC(
        pool5,
        'fc6',
        dim_in * 7 * 7,
        dim_fc6,
        weight_init=weight_init,
        bias_init=bias_init,
    )
    relu6 = model.Relu(fc6, 'fc6')
    drop6 = model.Dropout(relu6, 'drop6', ratio=0.5, is_test=not model.train)

    fc7 = model.FC(
        drop6,
        'fc7',
        dim_fc6,
        dim_fc7,
        weight_init=weight_init,
        bias_init=bias_init,
    )
    relu7 = model.Relu(fc7, 'fc7')
    drop7 = model.Dropout(relu7, 'drop7', ratio=0.5, is_test=not model.train)
    return drop7, dim_fc7, spatial_scale / 2.
