from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils
from detectron.modeling.ResNet import add_stage
from detectron.modeling.ResNet18 import add_stage as add_stage18
from detectron.ops.pcl import PCLOp

from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

# ---------------------------------------------------------------------------- #
# WSL outputs and losses
# ---------------------------------------------------------------------------- #


def add_wsl_outputs(model, blob_in, dim, prefix=''):
    """Add RoI classification and bounding box regression output ops."""
    if cfg.WSL.CONTEXT:
        fc8c, fc8d = add_wsl_context_outputs(model, blob_in, dim, prefix=prefix)
    else:
        # Box classification layer
        fc8c = model.FC(
            blob_in,
            prefix + 'fc8c',
            dim,
            model.num_classes - 1,
            weight_init=('XavierFill', {}),
            # weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )
        fc8d = model.FC(
            blob_in,
            prefix + 'fc8d',
            dim,
            model.num_classes - 1,
            weight_init=('XavierFill', {}),
            # weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )

    if cfg.WSL.CMIL and model.train:
        fc8c, fc8d = add_wsl_cmil(model, [fc8c, fc8d], dim, prefix=prefix)

    model.Softmax(fc8c, prefix + 'alpha_cls', axis=1)
    model.Transpose(fc8d, prefix + 'fc8d_t', axes=(1, 0))
    model.Softmax(prefix + 'fc8d_t', prefix + 'alpha_det_t', axis=1)
    model.Transpose(prefix + 'alpha_det_t', prefix + 'alpha_det', axes=(1, 0))
    model.net.Mul([prefix + 'alpha_cls', prefix + 'alpha_det'],
                  prefix + 'rois_pred')

    if not model.train:  # == if test
        # Add BackGround predictions
        model.net.Split(
            prefix + 'rois_pred', [prefix + 'rois_bg_pred', prefix + 'notuse'],
            split=[1, model.num_classes - 2],
            axis=1)
        model.net.Concat(
            [prefix + 'rois_bg_pred', prefix + 'rois_pred'],
            [prefix + 'cls_prob', prefix + 'cls_prob_concat_dims'],
            axis=1)

    if cfg.WSL.CONTEXT:
        blob_in = blob_in[0]
        dim = dim

    if cfg.WSL.CMIL:
        add_wsl_cmil_outputs(model, blob_in, dim, prefix=prefix)
    elif cfg.WSL.OICR :
        add_wsl_oicr_outputs(model, blob_in, dim, prefix=prefix)
    elif cfg.WSL.PCL:
        add_wsl_pcl_outputs(model, blob_in, dim, prefix=prefix)


def add_wsl_cmil(model, blob_in, dim, prefix=''):

    old_prefix = prefix
    prefix = prefix + 'cmil_'

    fc8c, fc8d = blob_in

    model.Softmax(fc8c, prefix + 'alpha_cls', axis=1)
    model.Transpose(fc8d, prefix + 'fc8d_t', axes=(1, 0))
    model.Softmax(prefix + 'fc8d_t', prefix + 'alpha_det_t', axis=1)
    model.Transpose(prefix + 'alpha_det_t', prefix + 'alpha_det', axes=(1, 0))
    model.net.Mul([prefix + 'alpha_cls', prefix + 'alpha_det'],
                  prefix + 'rois_pred')
    model.net.ReduceMax(prefix + 'rois_pred', prefix + 'rois_obn_score', axes=[1], keepdims=True)

    model.net.RoIIoU([old_prefix + 'rois'], [prefix + 'rois_iou'])
    model.net.RoIMerge([prefix + 'rois_obn_score', prefix + 'rois_iou',
                        fc8c, fc8d],
                       [prefix + 'fc8c', prefix + 'fc8d',
                        prefix + 'I', prefix + 'IC'],
                       display=int(1280 / cfg.NUM_GPUS),
                       size_epoch=cfg.WSL.SIZE_EPOCH,
                       max_epoch=int(cfg.SOLVER.MAX_ITER / cfg.WSL.SIZE_EPOCH),
                       )

    return prefix + 'fc8c', prefix + 'fc8d'


def add_wsl_cmil_outputs(model, blob_in, dim, prefix=''):
    K = 2
    for k in range(1, K+1):
        # Box classification layer
        model.FC(
            blob_in,
            prefix + 'cls_score' + str(k),
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )

    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        all_cls_prob = []
        for k in range(1, K+1):
            cls_prob = model.Softmax(prefix + 'cls_score' + str(k),
                                     prefix + 'cls_prob' + str(k),
                                     axis=1)
            all_cls_prob += [cls_prob]
        model.net.Mean(all_cls_prob, prefix + 'cls_prob')


def add_wsl_oicr_outputs(model, blob_in, dim, prefix=''):
    K = 3
    for k in range(1, K+1):
        # Box classification layer
        model.FC(
            blob_in,
            prefix + 'cls_score' + str(k),
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )

    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        all_cls_prob = []
        for k in range(1, K+1):
            cls_prob = model.Softmax(prefix + 'cls_score' + str(k),
                                     prefix + 'cls_prob' + str(k),
                                     axis=1)
            all_cls_prob += [cls_prob]
        model.net.Mean(all_cls_prob, prefix + 'cls_prob')


def add_wsl_pcl_outputs(model, blob_in, dim, prefix=''):
    K = 3
    for k in range(1, K+1):
        # Box classification layer
        model.FC(
            blob_in,
            prefix + 'cls_score' + str(k),
            dim,
            model.num_classes,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0),
        )

        model.Softmax(prefix + 'cls_score' + str(k),
                      prefix + 'cls_prob' + str(k),
                      axis=1)

    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        all_cls_prob = []
        for k in range(1, K+1):
            all_cls_prob += [prefix + 'cls_prob' + str(k)]
        model.net.Mean(all_cls_prob, prefix + 'cls_prob')


def add_wsl_context_outputs(model, blobs_in, dim, prefix=''):
    """Add RoI classification and bounding box regression output ops."""
    # Box classification layer
    fc8c = model.FC(
        blobs_in[0],
        prefix + 'fc8c',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    fc8d_f = model.FC(
        blobs_in[1],
        prefix + 'fc8d_frame',
        dim,
        model.num_classes - 1,
        weight_init=('XavierFill', {}),
        bias_init=const_fill(0.0))
    fc8d_c = model.net.FC(
        [blobs_in[2],
         prefix + 'fc8d_frame_w',
         prefix + 'fc8d_frame_b'],
        prefix + 'fc8d_context')
    fc8d = model.net.Sub([fc8d_f, fc8d_c], prefix + 'fc8d')

    return fc8c, fc8d



def add_cls_pred(in_blob, out_blob, model, prefix=''):
    assert cfg.TRAIN.IMS_PER_BATCH == 1, 'Only support one image per GPU'

    if False:
        model.net.RoIScoreReshape([in_blob, 'rois'],
                                  in_blob + '_reshape',
                                  num_classes=model.num_classes - 1,
                                  batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                  rois_size=cfg.TRAIN.BATCH_SIZE_PER_IM)
        model.net.RoIScorePool(
            in_blob + '_reshape', out_blob, num_classes=model.num_classes - 1)

        return

    model.net.ReduceSum(in_blob, out_blob, axes=[0], keepdims=True)


def add_center_loss(label_blob, pred_blob, feature_blob, feature_dims, model):
    CF = model.create_param(
        param_name='center_feature',
        initializer=initializers.Initializer("GaussianFill"),
        tags=ParameterTags.COMPUTED_PARAM,
        shape=[
            model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER, feature_dims
        ],
    )

    dCF = model.create_param(
        param_name='center_feature_g',
        initializer=initializers.Initializer("ConstantFill", value=0.0),
        # tags=ParameterTags.COMPUTED_PARAM,
        shape=[
            model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER, feature_dims
        ],
    )

    ndCF = model.create_param(
        param_name='center_feature_n_u',
        initializer=initializers.Initializer("ConstantFill", value=0.0),
        # tags=ParameterTags.COMPUTED_PARAM,
        shape=[model.num_classes - 1, cfg.WSL.CENTER_LOSS_NUMBER],
    )

    if cfg.WSL.CPG or cfg.WSL.CSC:
        input_blobs = [
            label_blob, pred_blob, feature_blob, CF, dCF, ndCF, 'cpg'
        ]
    else:
        input_blobs = [label_blob, pred_blob, feature_blob, CF, dCF, ndCF]

    output_blobs = ['loss_center', 'D', 'S']

    loss_center, D, S = model.net.CenterLoss(
        input_blobs,
        output_blobs,
        max_iter=cfg.WSL.CSC_MAX_ITER,
        top_k=cfg.WSL.CENTER_LOSS_TOP_K,
        display=int(1280 / cfg.NUM_GPUS),
        update=int(128 / cfg.NUM_GPUS))

    loss_gradients = get_loss_gradients_weighted(model, [loss_center], 0.4096)
    model.AddLosses(['loss_center'])

    return loss_gradients


def add_min_entropy_loss(model, pred, label, loss, cpg=None):
    in_blobs = [pred, label]
    if cpg:
        in_blobs.append(cpg)
    out_blobs = [loss]
    loss_entropy = model.net.MinEntropyLoss(in_blobs, out_blobs)

    loss_gradients = get_loss_gradients_weighted(model, [loss_entropy], 0.1)
    model.AddLosses([loss])

    return loss_gradients


def add_cross_entropy_loss(model, pred, label, loss, weight=None, cpg=None):
    in_blob = [pred, label]
    if cpg:
        in_blob.append(cpg)
    out_blob = [loss]

    if weight:
        in_blob.insert(2, weight)
        model.net.WeightedCrossEntropyWithLogits(in_blob, out_blob, is_mean=cfg.WSL.MEAN_LOSS)
    else:
        model.net.CrossEntropyWithLogits(in_blob, out_blob, is_mean=cfg.WSL.MEAN_LOSS)


def add_csc_loss(model,
                 cpg_blob='cpg',
                 cls_prob_blob='cls_prob',
                 rois_pred_blob='rois_pred',
                 rois_blob='rois',
                 loss_weight=1.0,
                 csc_layer='CSC',
                 prefix='',
                 **kwargs):
    csc_func = getattr(model.net, csc_layer)
    csc_args = {}
    csc_args['tau'] = cfg.WSL.CPG_TAU
    csc_args['max_iter'] = cfg.WSL.CSC_MAX_ITER
    # csc_args['debug_info'] = cfg.WSL.DEBUG
    csc_args['fg_threshold'] = cfg.WSL.CSC_FG_THRESHOLD
    csc_args['mass_threshold'] = cfg.WSL.CSC_MASS_THRESHOLD
    csc_args['density_threshold'] = cfg.WSL.CSC_DENSITY_THRESHOLD
    csc_args.update(kwargs)
    csc, labels_oh_pos, labels_oh_neg = csc_func(
        [cpg_blob, 'labels_oh', cls_prob_blob, rois_blob],
        [prefix + 'csc', prefix + 'labels_oh_pos', prefix + 'labels_oh_neg'],
        **csc_args)

    model.net.CSCConstraint([rois_pred_blob, csc],
                            [prefix + 'rois_pred_pos', prefix + 'csc_pos'],
                            polar=True)
    model.net.CSCConstraint([rois_pred_blob, csc],
                            [prefix + 'rois_pred_neg', prefix + 'csc_neg'],
                            polar=False)

    add_cls_pred(prefix + 'rois_pred_pos', prefix + 'cls_prob_pos', model)
    add_cls_pred(prefix + 'rois_pred_neg', prefix + 'cls_prob_neg', model)

    weight = None

    add_cross_entropy_loss(
        model,
        prefix + 'cls_prob_pos',
        prefix + 'labels_oh_pos',
        prefix + 'cross_entropy_pos',
        cpg=cpg_blob,
        weight=weight)

    add_cross_entropy_loss(
        model,
        prefix + 'cls_prob_neg',
        prefix + 'labels_oh_neg',
        prefix + 'cross_entropy_neg',
        cpg=cpg_blob,
        weight=weight)

    loss_cls_pos = model.net.AveragedLoss([prefix + 'cross_entropy_pos'],
                                          [prefix + 'loss_cls_pos'])
    loss_cls_neg = model.net.AveragedLoss([prefix + 'cross_entropy_neg'],
                                          [prefix + 'loss_cls_neg'])

    # loss_gradients = blob_utils.get_loss_gradients(
    # model, [loss_cls_pos, loss_cls_neg])
    loss_gradients = get_loss_gradients_weighted(
        model, [loss_cls_pos, loss_cls_neg], loss_weight)
    model.Accuracy([prefix + 'cls_prob_pos', 'labels_int32'],
                   prefix + 'accuracy_cls_pos')
    # model.Accuracy(['cls_prob_neg', 'labels_int32'], 'accuracy_cls_neg')
    model.AddLosses([prefix + 'loss_cls_pos', prefix + 'loss_cls_neg'])
    # model.AddMetrics(['accuracy_cls_pos', 'accuracy_cls_neg'])
    model.AddMetrics([prefix + 'accuracy_cls_pos'])

    return loss_gradients


def add_wsl_losses(model, prefix=''):
    add_cls_pred(prefix + 'rois_pred', prefix + 'cls_prob', model, prefix='')
    classes_weight = None

    cpg = None
    if cfg.WSL.CPG or cfg.WSL.CSC:
        cpg_args = {}
        cpg_args['tau'] = cfg.WSL.CPG_TAU
        cpg_args['max_iter'] = max(cfg.WSL.CPG_MAX_ITER, cfg.WSL.CSC_MAX_ITER)
        # cpg_args['debug_info'] = cfg.WSL.DEBUG
        cpg_args['cpg_net_name'] = model.net.Proto().name + '_cpg'
        cpg_args['pred_blob_name'] = cfg.WSL.CPG_PRE_BLOB
        cpg_args['data_blob_name'] = cfg.WSL.CPG_DATA_BLOB

        model.net.CPG(['labels_oh', prefix + 'cls_prob'], ['cpg_raw'],
                      **cpg_args)
        model.net.CPGScale(['cpg_raw', 'labels_oh', prefix + 'cls_prob'],
                           'cpg',
                           tau=cfg.WSL.CPG_TAU)
        cpg = 'cpg'

    if cfg.WSL.CSC:
        if not cfg.MODEL.MASK_ON or True:
            loss_gradients = add_csc_loss(
                model,
                'cpg',
                prefix + 'cls_prob',
                prefix + 'rois_pred',
                prefix + 'rois',
                loss_weight=1.0,
                prefix='')
        else:
            loss_gradients = {}
    else:
        add_cross_entropy_loss(
            model,
            prefix + 'cls_prob',
            'labels_oh',
            prefix + 'cross_entropy',
            weight=classes_weight,
            cpg=cpg)
        loss_cls = model.net.AveragedLoss([prefix + 'cross_entropy'],
                                          [prefix + 'loss_cls'])

        loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls])
        model.Accuracy([prefix + 'cls_prob', 'labels_int32'],
                       prefix + 'accuracy_cls')
        model.AddLosses([prefix + 'loss_cls'])
        model.AddMetrics(prefix + 'accuracy_cls')

    if cfg.WSL.CENTER_LOSS:
        center_dim = 4096
        rois_pred = prefix + 'rois_pred'

        loss_gradients_center = add_center_loss(
            'labels_oh', rois_pred, prefix + 'drop7', center_dim, model)
        loss_gradients.update(loss_gradients_center)

    if cfg.WSL.MIN_ENTROPY_LOSS:
        loss_gradients_ME = add_min_entropy_loss(
            model,
            prefix + 'rois_pred',
            'labels_oh',
            prefix + 'loss_entropy',
            cpg=cpg)
        loss_gradients.update(loss_gradients_ME)


    if cfg.WSL.CMIL:
        loss_gradients_cmil = add_cmil_losses(model, prefix)
        loss_gradients.update(loss_gradients_cmil)
    elif cfg.WSL.OICR:
        loss_gradients_oicr = add_oicr_losses(model, prefix)
        loss_gradients.update(loss_gradients_oicr)
    elif cfg.WSL.PCL:
        loss_gradients_pcl = add_pcl_losses(model, prefix)
        loss_gradients.update(loss_gradients_pcl)

    return loss_gradients


def add_cmil_losses(model, prefix=''):
    loss_gradients = {}

    model.net.RoIIoU([prefix + 'rois'], [prefix + 'rois_iou'])

    import uuid
    uu = uuid.uuid4().int % 10000

    K = 2
    for k in range(1, K+1):
        if k == 1:
            input_blobs = [prefix + 'cmil_rois_pred',
                           prefix + 'rois_iou',
                           'labels_oh',
                           prefix + 'cls_prob']
        else:
            input_blobs = [prefix + 'cls_prob' + str(k-1),
                           prefix + 'rois_iou',
                           'labels_oh',
                           prefix + 'cls_prob']

        model.net.RoILabel(input_blobs,
                           [prefix + 'rois_labels_int32' + str(k),
                            prefix + 'rois_weight' + str(k)],
                           display=int(1280 / cfg.NUM_GPUS),
                           uuid=uu,
                           fg_thresh=0.6,
                           bg_thresh_hi=0.4,
                           bg_thresh_lo=0.1,
                           num_pos=32,
                           num_neg=96,
                           )

        cls_prob, loss_cls = model.net.SoftmaxWithLossN(
            [prefix + 'cls_score' + str(k),
             prefix + 'rois_labels_int32' + str(k),
             prefix + 'rois_weight' + str(k)],
            [prefix + 'cls_prob' + str(k),
             prefix + 'loss_cls' + str(k)],
            # scale=model.GetLossScale(),
        )

        if cfg.WSL.MEAN_LOSS:
            lg = blob_utils.get_loss_gradients(model, [loss_cls])
        else:
            lg = get_loss_gradients_weighted(model, [loss_cls], 1. * (cfg.MODEL.NUM_CLASSES -1))
        loss_gradients.update(lg)
        model.Accuracy([prefix + 'cls_prob' + str(k),
                        prefix + 'rois_labels_int32' + str(k)],
                       prefix + 'accuracy_cls' + str(k))
        model.AddLosses([prefix + 'loss_cls' + str(k)])
        model.AddMetrics(prefix + 'accuracy_cls' + str(k))

    return loss_gradients


def add_oicr_losses(model, prefix=''):
    loss_gradients = {}

    model.net.RoIIoU([prefix + 'rois'], [prefix + 'rois_iou'])

    import uuid
    uu = uuid.uuid4().int % 10000

    K = 3
    for k in range(1, K+1):
        if k == 1:
            input_blobs = [prefix + 'rois_pred',
                           prefix + 'rois_iou',
                           'labels_oh',
                           prefix + 'cls_prob']
        else:
            input_blobs = [prefix + 'cls_prob' + str(k-1),
                           prefix + 'rois_iou',
                           'labels_oh',
                           prefix + 'cls_prob']

        model.net.RoILabel(input_blobs,
                           [prefix + 'rois_labels_int32' + str(k),
                            prefix + 'rois_weight' + str(k)],
                           display=int(1280 / cfg.NUM_GPUS),
                           uuid=uu,
                           )

        cls_prob, loss_cls = model.net.SoftmaxWithLossN(
            [prefix + 'cls_score' + str(k),
             prefix + 'rois_labels_int32' + str(k),
             prefix + 'rois_weight' + str(k)],
            [prefix + 'cls_prob' + str(k),
             prefix + 'loss_cls' + str(k)],
            # scale=model.GetLossScale(),
        )

        if cfg.WSL.MEAN_LOSS:
            lg = blob_utils.get_loss_gradients(model, [loss_cls])
        else:
            lg = get_loss_gradients_weighted(model, [loss_cls], 1. * (cfg.MODEL.NUM_CLASSES -1))
        loss_gradients.update(lg)
        model.Accuracy([prefix + 'cls_prob' + str(k),
                        prefix + 'rois_labels_int32' + str(k)],
                       prefix + 'accuracy_cls' + str(k))
        model.AddLosses([prefix + 'loss_cls' + str(k)])
        model.AddMetrics(prefix + 'accuracy_cls' + str(k))

    return loss_gradients


def add_pcl_losses(model, prefix=''):
    loss_gradients = {}

    K = 3
    for k in range(1, K+1):
        if k == 1:
            input_blobs = [prefix + 'rois',
                           prefix + 'rois_pred',
                           'labels_oh',
                           prefix + 'cls_prob' + str(k),
                           prefix + 'cls_prob',
                           ]
        else:
            input_blobs = [prefix + 'rois',
                           prefix + 'cls_prob' + str(k-1),
                           'labels_oh',
                           prefix + 'cls_prob' + str(k),
                           prefix + 'cls_prob',
                           ]

        output_blobs = [prefix + 'labels' + str(k),
                        prefix + 'cls_loss_weights' + str(k),
                        prefix + 'gt_assignment' + str(k),
                        prefix + 'pc_labels' + str(k),
                        prefix + 'pc_probs' + str(k),
                        prefix + 'pc_count' + str(k),
                        prefix + 'img_cls_loss_weights' + str(k),
                        prefix + 'im_labels_real' + str(k),
                        ]

        name = 'PCL:' + str(k)
        model.net.Python(PCLOp().forward)(
            input_blobs, output_blobs, name=name
        )

        loss_cls = model.net.PCLLoss(
            [prefix + 'cls_prob' + str(k),] + output_blobs,
            [prefix + 'loss_cls' + str(k)],
        )

        lg = blob_utils.get_loss_gradients(model, [loss_cls])
        loss_gradients.update(lg)
        model.AddLosses([prefix + 'loss_cls' + str(k)])

    return loss_gradients


def get_loss_gradients_weighted(model, loss_blobs, loss_weight):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(
            b, [b + '_grad'], value=1.0 * loss_weight)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    roi_feat = model.net.RoIFeatureBoost([roi_feat, prefix + 'obn_scores'], roi_feat)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        roi_feat = model.StopGradient(roi_feat, roi_feat)

    l = model.FC(roi_feat, prefix + 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', hidden_dim, hidden_dim)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, hidden_dim


def add_VGG16_roi_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale, prefix=prefix)
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, 4096)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', 4096, 4096)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, 4096


def add_VGG16_roi_context_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    _, _ = model.net.RoIContext([prefix + 'rois', 'data'],
                                [prefix + 'rois_frame', prefix + 'rois_context'])

    blobs_out = []
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    # origin roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, 4096)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', 4096, 4096)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    blobs_out += [l]

    # frame roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat_frame',
        blob_rois=prefix + 'rois_frame',
        method='RoILoopPool',
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.net.FC([l, prefix + 'fc6_w', prefix + 'fc6_b'], prefix + 'fc6_frame')
    l = model.Relu(l, prefix + 'fc6_frame')
    l = DropoutIfTraining(model, l, prefix + 'drop6_frame', 0.5)
    l = model.net.FC([l, prefix + 'fc7_w', prefix + 'fc7_b'], prefix + 'fc7_frame')
    l = model.Relu(l, prefix + 'fc7_frame')
    l = DropoutIfTraining(model, l, prefix + 'drop7_frame', 0.5)

    blobs_out += [l]

    # context roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat_context',
        blob_rois=prefix + 'rois_context',
        method='RoILoopPool',
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.net.FC([l, prefix + 'fc6_w', prefix + 'fc6_b'], prefix + 'fc6_context')
    l = model.Relu(l, prefix + 'fc6_context')
    l = DropoutIfTraining(model, l, prefix + 'drop6_context', 0.5)
    l = model.net.FC([l, prefix + 'fc7_w', prefix + 'fc7_b'], prefix + 'fc7_context')
    l = model.Relu(l, prefix + 'fc7_context')
    l = DropoutIfTraining(model, l, prefix + 'drop7_context', 0.5)

    blobs_out += [l]

    return blobs_out, 4096


def add_ResNet_roi_0fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""

    if cfg.WSL.CONTEXT:
        return add_ResNet_roi_context_0fc_head(model, blob_in, dim_in,
                                              spatial_scale)

    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois= prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    return l, dim_in * roi_size * roi_size

    s = model.AveragePool(l, prefix + 'roi_feat_pool', kernel=roi_size)

    return s, dim_in


def add_ResNet_roi_context_0fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""

    blobs_out = []

    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois= prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    blobs_out += [l]
    blobs_out += [l]
    blobs_out += [l]

    return blobs_out, dim_in * roi_size * roi_size

    s = model.AveragePool(l, prefix + 'roi_feat_pool', kernel=roi_size)

    return s, dim_in


def add_ResNet_roi_1fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    """Add a ReLU MLP with two hidden layers."""

    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois= prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)

    return l, hidden_dim


def add_ResNet_roi_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    """Add a ReLU MLP with two hidden layers."""
    if cfg.WSL.CONTEXT:
        return add_ResNet_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale, prefix=prefix)

    if len(cfg.WSL.MLP_HEAD_DIM) == 2:
        hidden_dim6 = cfg.WSL.MLP_HEAD_DIM[0]
        hidden_dim7 = cfg.WSL.MLP_HEAD_DIM[1]
    else:
        hidden_dim6 = cfg.FAST_RCNN.MLP_HEAD_DIM
        hidden_dim7 = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois= prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l = model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, hidden_dim6)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', hidden_dim6, hidden_dim7)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, hidden_dim7


def add_ResNet_roi_context_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    _, _ = model.net.RoIContext([prefix + 'rois', 'data'],
                                [prefix + 'rois_frame', prefix + 'rois_context'])

    blobs_out = []
    if len(cfg.WSL.MLP_HEAD_DIM) == 2:
        hidden_dim6 = cfg.WSL.MLP_HEAD_DIM[0]
        hidden_dim7 = cfg.WSL.MLP_HEAD_DIM[1]
    else:
        hidden_dim6 = cfg.FAST_RCNN.MLP_HEAD_DIM
        hidden_dim7 = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    # origin roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l_ori = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # frame roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat_frame',
        blob_rois=prefix + 'rois_frame',
        method='RoILoopPool',
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l_fra = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # context roi
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat_context',
        blob_rois=prefix + 'rois_context',
        method='RoILoopPool',
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l_con = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        l_ori = model.StopGradient(l_ori, l_ori)
        l_fra = model.StopGradient(l_fra, l_fra)
        l_con = model.StopGradient(l_con, l_con)

    if cfg.FPN.FPN_ON:
        dim_in = dim_in[0]
    l = model.FC(l_ori, prefix + 'fc6', dim_in * roi_size * roi_size, hidden_dim6)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', hidden_dim6, hidden_dim7)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    blobs_out += [l]

    l = model.net.FC([l_fra, prefix + 'fc6_w', prefix + 'fc6_b'], prefix + 'fc6_frame')
    l = model.Relu(l, prefix + 'fc6_frame')
    l = DropoutIfTraining(model, l, prefix + 'drop6_frame', 0.5)
    l = model.net.FC([l, prefix + 'fc7_w', prefix + 'fc7_b'], prefix + 'fc7_frame')
    l = model.Relu(l, prefix + 'fc7_frame')
    l = DropoutIfTraining(model, l, prefix + 'drop7_frame', 0.5)

    blobs_out += [l]

    l = model.net.FC([l_con, prefix + 'fc6_w', prefix + 'fc6_b'], prefix + 'fc6_context')
    l = model.Relu(l, prefix + 'fc6_context')
    l = DropoutIfTraining(model, l, prefix + 'drop6_context', 0.5)
    l = model.net.FC([l, prefix + 'fc7_w', prefix + 'fc7_b'], prefix + 'fc7_context')
    l = model.Relu(l, prefix + 'fc7_context')
    l = DropoutIfTraining(model, l, prefix + 'drop7_context', 0.5)

    blobs_out += [l]

    return blobs_out, hidden_dim7


def add_roi_Xconv_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
                                               'roi_feat_boost')

    current = roi_feat_boost
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    current = model.AveragePool(current, 'head_pool', kernel=roi_size)

    return current, dim_in


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    roi_feat_boost = model.net.RoIFeatureBoost([roi_feat, 'obn_scores'],
                                               'roi_feat_boost')

    current = roi_feat_boost
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current,
            'head_conv' + str(i + 1),
            dim_in,
            hidden_dim,
            3,
            group_gn=get_group_gn(hidden_dim),
            stride=1,
            pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {
                'value': 0.
            }))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    # return 'fc6', fc_dim

    DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    return 'drop6', fc_dim


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    l = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(model, 'res5', l, 3, dim_in, 2048,
                          dim_bottleneck * 8, 1, stride_init)
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 2048


def add_ResNet18_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    l = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage18(model, 'res5', l, 2, dim_in, 512,
                          0, cfg.RESNETS.RES5_DILATION, stride_init)
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 512


def add_AlexNet_roi_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale)
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, 4096)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', 4096, 4096)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, 4096


def add_VGG_CNN_roi_2fc_head(model, blob_in, dim_in, spatial_scale, prefix=''):
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    if cfg.WSL.CONTEXT:
        return add_VGG16_roi_context_2fc_head(model, blob_in, dim_in,
                                              spatial_scale)
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    l = model.RoIFeatureTransform(
        blob_in,
        prefix + 'roi_feat',
        blob_rois=prefix + 'rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    l = model.net.RoIFeatureBoost([l, prefix + 'obn_scores'], l)

    # save memory
    if cfg.TRAIN.FREEZE_CONV_BODY:
        model.StopGradient(l, l)

    l = model.FC(l, prefix + 'fc6', dim_in * roi_size * roi_size, 4096)
    l = model.Relu(l, prefix + 'fc6')
    l = DropoutIfTraining(model, l, prefix + 'drop6', 0.5)
    l = model.FC(l, prefix + 'fc7', 4096, hidden_dim)
    l = model.Relu(l, prefix + 'fc7')
    l = DropoutIfTraining(model, l, prefix + 'drop7', 0.5)

    return l, hidden_dim


def DropoutIfTraining(model, blob_in, blob_out, dropout_rate):
    """Add dropout to blob_in if the model is in training mode and
    dropout_rate is > 0."""
    if model.train and dropout_rate > 0:
        blob_out = model.Dropout(
            blob_in, blob_out, ratio=dropout_rate, is_test=False)
        return blob_out
    else:
        return blob_in
