from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import time
import copy
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2
import logging
import caffe2.python._import_c_extension as C

from caffe2.python.memonger import verify_graph_equality, verify_inplace_blobs

log = logging.getLogger("memonger")
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ["defined", "used", "size"])


def share_freeze_blobs(
        net,
        namescope,
):

    log.warn("NOTE: Executing memonger to optimize gradient memory")

    # Collect ops that have something to do with gradients
    if namescope != "" and not namescope.endswith("/"):
        namescope += "/"

    netproto = copy.deepcopy(net.Proto())
    new_net = copy.deepcopy(net)
    activations = []
    external_input = set(new_net.Proto().external_input)
    external_output = set(new_net.Proto().external_output)

    start_idx = -1
    end_idx = -1

    # ops
    for idx, op in enumerate(new_net._net.op):
        # print(op)
        if namescope not in op.input[0]:
            continue
        if op.type == 'Conv' and start_idx < 0:
            start_idx = idx
        if op.type == 'StopGradient':
            end_idx = idx

    # print(namescope, 'start_idx: ', start_idx, ' end_idx: ', end_idx)

    # Hacky way to get activations, think of a better way
    for idx, op in enumerate(new_net._net.op[start_idx:end_idx]):
        if namescope not in op.input[0]:
            continue
        for b in op.output:
            if b not in external_output:
                activations.append(b)

    # print('activations: ', activations)

    used_activations = []
    for a in activations:
        if a in used_activations:
            continue
        share_pool = [
            namescope + '_shared_' + str(i) for i in range(1000, 10000)
        ]
        # print(a)
        first_idx = -1
        for idx, op in enumerate(new_net._net.op):
            if namescope not in op.input[0]:
                continue
            if a in list(op.input) + list(op.output):
                first_idx = idx
                break

        assert first_idx >= 0, first_idx

        for idx, op in enumerate(new_net._net.op[first_idx:]):
            if namescope not in op.input[0]:
                continue
            for b in list(op.input) + list(op.output):
                if b in share_pool:
                    share_pool.remove(b)

        for idx, op in enumerate(new_net._net.op):
            if namescope not in op.input[0]:
                continue
            op_input = copy.deepcopy(op.input)
            is_found = False
            for i, b in enumerate(op_input):
                if a == b:
                    op_input[i] = share_pool[-1]
                    is_found = True
            if is_found:
                del new_net._net.op[idx].input[:]
                new_net._net.op[idx].input.extend(op_input)

            op_output = copy.deepcopy(op.output)
            is_found = False
            for i, b in enumerate(op_output):
                if a == b:
                    op_output[i] = share_pool[-1]
                    is_found = True
            if is_found:
                del new_net._net.op[idx].output[:]
                new_net._net.op[idx].output.extend(op_output)

        used_activations.append(a)

    assert verify_graph_equality(net.Proto(), new_net.Proto()), \
        "Memonger graph is not equal to original."
    assert verify_inplace_blobs(net.Proto(), new_net.Proto()), \
        "Inplace assignments differ in memonger net."

    share_pool = [
        namescope + '_shared_' + str(i) for i in range(1000, 10000)
    ]
    share_pool_used = {}
    for idx, op in enumerate(new_net._net.op):
        if namescope not in op.input[0]:
            continue
        for b in list(op.input) + list(op.output):
            if b in share_pool:
                share_pool_used[b] = idx

    for idx, op in enumerate(new_net._net.op[end_idx:]):
        if namescope not in op.input[0]:
            continue
        for b in list(op.input) + list(op.output):
            if b in share_pool_used.keys():
                share_pool_used.pop(b)

    ops = list(new_net._net.op)
    for inp in share_pool_used.keys():
        # print('free: ', inp)
        # new_net.Free([inp], [inp])
        
        ops.insert(share_pool_used[inp] + 1, core.CreateOperator("Free", [inp], [inp]))
    del new_net._net.op[:]
    new_net._net.op.extend(ops)

    return new_net.Proto()


def share_freeze_blobs_c2(
        net,
        namescope,
):

    log.warn("NOTE: Executing memonger to optimize gradient memory")

    # Collect ops that have something to do with gradients
    if namescope != "" and not namescope.endswith("/"):
        namescope += "/"

    netproto = copy.deepcopy(net.Proto())
    activations = []
    external_input = set(net.Proto().external_input)
    external_output = set(net.Proto().external_output)

    start_idx = -1
    end_idx = -1

    # ops
    for idx, op in enumerate(netproto.op):
        # print(op)
        if namescope not in op.input[0]:
            continue
        if op.type == 'Conv' and start_idx < 0:
            start_idx = idx
        if op.type == 'StopGradient':
            end_idx = idx

    print(namescope, 'start_idx: ', start_idx, ' end_idx: ', end_idx)

    # Hacky way to get activations, think of a better way
    for idx, op in enumerate(netproto.op[start_idx:end_idx]):
        for b in op.output:
            if b not in external_output:
                activations.append(b)

    print('activations: ', activations)

    share_pool = [namescope + '_shared_' + str(i) for i in range(1000, 10000)]
    map_pool = {}

    heads = [namescope + 'data']
    print('heads: ', heads)

    # Remove last activations, as they are usually accessed externally
    activations = set(activations[:-1])
    print('activations: ', activations)

    shared_blobs = activations
    dont_share_blobs = None
    blob_shapes = None
    op_indices = [
        index for index, op in enumerate(netproto.op[start_idx:end_idx + 2])
    ]

    print(op_indices)

    start_time = time.time()
    optim_str = C.memonger_compute_blob_recycling_for_dag(
        netproto.SerializeToString(), [str(s).encode('utf-8') for s in heads],
        op_indices, set(str(s).encode('utf-8') for s in shared_blobs),
        namescope.encode('utf-8'),
        set() if dont_share_blobs is None else dont_share_blobs,
        {} if blob_shapes is None else blob_shapes)

    optim = caffe2_pb2.NetDef()
    optim.ParseFromString(optim_str)
    assert verify_graph_equality(net.Proto(), optim), \
        "Memonger graph is not equal to original."
    assert verify_inplace_blobs(net.Proto(), optim), \
        "Inplace assignments differ in memonger net."
    return optim
