from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from caffe2.python import memonger
from caffe2.python import workspace

from caffe2.python import muji
import detectron.utils.c2 as c2_utils

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.modeling import model_builder_wsl


def create_cpg_net(train=True):
    logger = logging.getLogger(__name__)

    FREEZE_CONV_BODY = cfg.TRAIN.FREEZE_CONV_BODY
    FREEZE_AT = cfg.TRAIN.FREEZE_AT
    WSL_CSC = cfg.WSL.CSC
    CENTER_LOSS = cfg.WSL.CENTER_LOSS
    MIN_ENTROPY_LOSS = cfg.WSL.MIN_ENTROPY_LOSS
    MASK_ON = cfg.MODEL.MASK_ON
    EXECUTION_TYPE = cfg.MODEL.EXECUTION_TYPE

    cfg.immutable(False)
    cfg.TRAIN.FREEZE_CONV_BODY = False
    cfg.TRAIN.FREEZE_AT = 0
    cfg.WSL.CSC = False
    cfg.WSL.CENTER_LOSS = False
    cfg.WSL.MIN_ENTROPY_LOSS = False
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.EXECUTION_TYPE = b'simple'
    cfg.immutable(True)

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    for gpu_id in range(cfg.NUM_GPUS):
        logger.info('Building model: {}'.format('gpu_' + str(gpu_id) + '_' +
                                                cfg.MODEL.TYPE + '_cpg'))
        model = model_builder_wsl.create(
            'gpu_' + str(gpu_id) + '_' + cfg.MODEL.TYPE + '_cpg', train=train)
        # workspace.RunNetOnce(model.param_init_net)

        #-----------------------------------------------------------------------------------
        # logger.info(
        # 'Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
        # dump_proto_files(model, output_dir)

        if cfg.MEMONGER and False:
            start_op, end_op = OP_surgery_head(model, gpu_id)
            optimize_memory_cpg(model, gpu_id)
            OP_surgery_back(model, gpu_id, start_op, end_op)
            namescope = 'gpu_{}/'.format(gpu_id)
            model.net._net.op[0].input[
                0] = namescope + cfg.WSL.CPG_PRE_BLOB + '_grad'
            model.net._net.op[-1].output[
                -1] = namescope + cfg.WSL.CPG_DATA_BLOB + '_grad'
            # share_surgery(model, gpu_id)
        else:
            OP_surgery(model, gpu_id)
        Input_surgery(model, gpu_id)
        workspace.CreateBlob('gpu_' + str(gpu_id) + '/' +
                             cfg.WSL.CPG_PRE_BLOB + '_grad')
        optimize_memory_cpg(model, gpu_id)
        #-----------------------------------------------------------------------------------

        # nu.broadcast_parameters(model)
        workspace.CreateNet(model.net)

        logger.info('Outputs saved to: {:s}'.format(
            os.path.abspath(output_dir)))
        dump_proto_files(model, output_dir)

    cfg.immutable(False)
    cfg.TRAIN.FREEZE_CONV_BODY = FREEZE_CONV_BODY
    cfg.TRAIN.FREEZE_AT = FREEZE_AT
    cfg.WSL.CSC = WSL_CSC
    cfg.WSL.CENTER_LOSS = CENTER_LOSS
    cfg.WSL.MIN_ENTROPY_LOSS = MIN_ENTROPY_LOSS
    cfg.MODEL.MASK_ON = MASK_ON
    cfg.MODEL.EXECUTION_TYPE = EXECUTION_TYPE
    cfg.immutable(True)


def optimize_memory_cpg_new(model, device):
    """Save GPU memory through blob sharing."""
    # for device in range(cfg.NUM_GPUS):
    if device >= 0:
        namescope = 'gpu_{}/'.format(device)
        # it seem dont_share_blobs not working
        dont_share_blobs = set([
            namescope + cfg.WSL.CPG_PRE_BLOB + '_grad',
            namescope + cfg.WSL.CPG_DATA_BLOB + '_grad'
        ])
        # losses = [namescope + l for l in model.losses]
        heads = [namescope + cfg.WSL.CPG_PRE_BLOB + '_grad']
        import detectron.utils.cpg_memonger as cpg_memonger
        model.net._net = cpg_memonger.share_blobs(
            model.net,
            heads,
            namescope,
            dont_share_blobs=dont_share_blobs,
        )


def optimize_memory_cpg(model, device):
    namescope = 'gpu_{}/'.format(device)
    dont_free_blobs = set([
        namescope + cfg.WSL.CPG_PRE_BLOB + '_grad',
        namescope + cfg.WSL.CPG_DATA_BLOB + '_grad'
    ])
    import detectron.utils.cpg_memonger as cpg_memonger
    model.net._net = cpg_memonger.deep_release_blobs_when_used(
        model.net._net, dont_free_blobs)
    return

    namescope = 'gpu_{}/'.format(device)
    dont_share_blobs = set([
        namescope + cfg.WSL.CPG_PRE_BLOB + '_grad',
        namescope + cfg.WSL.CPG_DATA_BLOB + '_grad'
    ])
    import detectron.utils.cpg_memonger as cpg_memonger
    cpg_memonger.deep_share_blobs(
        model.net,
        namescope,
        dont_share_blobs=dont_share_blobs,
    )
    return

    optimize_memory_cpg_new(model, device)
    return
    """Save GPU memory through blob sharing."""
    # for device in range(cfg.NUM_GPUS):
    if device >= 0:
        namescope = 'gpu_{}/'.format(device)
        # it seem dont_share_blobs not working
        dont_share_blobs = set([
            namescope + cfg.WSL.CPG_PRE_BLOB + '_grad',
            namescope + cfg.WSL.CPG_DATA_BLOB + '_grad'
        ])
        losses = [namescope + l for l in model.losses]
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses,
            set(model.param_to_grad.values()),
            namescope,
            share_activations=cfg.MEMONGER_SHARE_ACTIVATIONS,
            dont_share_blobs=dont_share_blobs,
        )


def share_surgery(model, gpu_id):
    num_op = len(model.net._net.op)
    for i in range(num_op):
        for j in range(len(model.net._net.op[i].input)):
            if '_shared' in model.net._net.op[i].input[j]:
                model.net._net.op[i].input[
                    j] = model.net._net.op[i].input[j] + '_cpg'

        for j in range(len(model.net._net.op[i].output)):
            if '_shared' in model.net._net.op[i].output[j]:
                model.net._net.op[i].output[
                    j] = model.net._net.op[i].output[j] + '_cpg'


def OP_surgery_head(model, gpu_id):
    num_op = len(model.net._net.op)
    for i in range(num_op):
        if 'gpu_' + str(gpu_id) not in model.net._net.op[i].input[0]:
            continue
        if cfg.WSL.CPG_PRE_BLOB + '_grad' in model.net._net.op[i].input[0]:
            start_op = i
            break
    for i in reversed(range(num_op)):
        if 'gpu_' + str(gpu_id) not in model.net._net.op[i].input[0]:
            continue
        print(cfg.WSL.CPG_DATA_BLOB + '_grad', model.net._net.op[i].output[-1])
        if cfg.WSL.CPG_DATA_BLOB + '_grad' in model.net._net.op[i].output[-1]:
            end_op = i + 1
            break
    return start_op, end_op


def OP_surgery_back(model, gpu_id, start_op, end_op):
    new_op = model.net._net.op[start_op:end_op]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def OP_surgery(model, gpu_id):
    num_op = len(model.net._net.op)
    for i in range(num_op):
        if 'gpu_' + str(gpu_id) not in model.net._net.op[i].input[0]:
            continue
        if cfg.WSL.CPG_PRE_BLOB + '_grad' in model.net._net.op[i].input[0]:
            start_op = i
            break
    for i in reversed(range(num_op)):
        if 'gpu_' + str(gpu_id) not in model.net._net.op[i].input[0]:
            continue
        if cfg.WSL.CPG_DATA_BLOB + '_grad' in model.net._net.op[i].output[-1]:
            end_op = i + 1
            break
    new_op = model.net._net.op[start_op:end_op]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def Input_surgery(model, gpu_id):
    num_op = len(model.net._net.op)

    all_input = []
    all_output = []
    for i in range(num_op):
        if 'gpu_' + str(gpu_id) not in model.net._net.op[i].input[0]:
            continue
        all_input.extend(model.net._net.op[i].input[:])
        all_output.extend(model.net._net.op[i].output[:])

    external_input = []
    for blob in all_input:
        if blob not in all_output:
            external_input.append(blob)

    new_external_input = []
    num_external_input = len(model.net._net.external_input)
    for i in range(num_external_input):
        # if 'gpu_' + str(gpu_id) not in model.net._net.external_input[i]:
        # continue
        # if 'momentum' in model.net._net.external_input[i]:
        # continue
        # if 'acmgrad' in model.net._net.external_input[i]:
        # continue
        # if '_w' in model.net._net.external_input[i]:
        # new_external_input.append(model.net._net.external_input[i])
        # if '_b' in model.net._net.external_input[i]:
        # new_external_input.append(model.net._net.external_input[i])
        if model.net._net.external_input[i] in all_input:
            new_external_input.append(model.net._net.external_input[i])

    num_external_input = len(external_input)
    for i in range(num_external_input):
        if external_input[i] in new_external_input:
            continue
        new_external_input.append(external_input[i])

    # new_external_input.extend(external_input)
    del model.net._net.external_input[:]
    model.net._net.external_input.extend(new_external_input)


def Center_loss_surgery(model):
    # broadcast parameters
    blobs = [
        'gpu_' + str(gpu_id) + '/center_feature'
        for gpu_id in range(cfg.NUM_GPUS)
    ]

    data = workspace.FetchBlob(blobs[0])
    for i, p in enumerate(blobs[1:]):
        with c2_utils.CudaScope(i + 1):
            workspace.FeedBlob(p, data)

    # sync parameters
    with c2_utils.CudaScope(0):
        gradients = [
            'gpu_' + str(gpu_id) + '/center_feature_g'
            for gpu_id in range(cfg.NUM_GPUS)
        ]
        if cfg.USE_NCCL:
            model.net.NCCLAllreduce(gradients, gradients)
        else:
            muji.Allreduce(model.net, gradients, reduced_affix='')

    with c2_utils.CudaScope(0):
        gradients = [
            'gpu_' + str(gpu_id) + '/center_feature_n_u'
            for gpu_id in range(cfg.NUM_GPUS)
        ]
        if cfg.USE_NCCL:
            model.net.NCCLAllreduce(gradients, gradients)
        else:
            muji.Allreduce(model.net, gradients, reduced_affix='')


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, model.net.Proto().name), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(
            os.path.join(output_dir,
                         model.param_init_net.Proto().name), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))
