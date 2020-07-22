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

log = logging.getLogger("memonger")
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ["defined", "used", "size"])


def share_blobs(
        net,
        heads,
        namescope,
        dont_share_blobs=None,
        blob_shapes=None,
):
    external_input = set(net.Proto().external_input)

    def is_new_blob(b):
        name = str(b)
        # Note: need to look at _{namescope} pattern as it matches
        # to handle the auto-split gradients
        return b not in external_input and (name.startswith(namescope) or
                                            name.startswith("_" + namescope))

    log.warn("NOTE: Executing memonger to optimize gradient memory")

    # Collect ops that have something to do with gradients
    if namescope != "" and not namescope.endswith("/"):
        namescope += "/"

    netproto = copy.deepcopy(net.Proto())

    # ops
    shared_op_indices = []
    for idx, op in enumerate(netproto.op):
        shared_op_indices.append(idx)

    shared_blobs = set()
    for op in net.Proto().op:
        for b in list(op.input) + list(op.output):
            if is_new_blob(b):
                shared_blobs.add(b)
    print(external_input)
    print(shared_blobs)
    start_time = time.time()
    optim_str = C.memonger_compute_blob_recycling_for_dag(
        netproto.SerializeToString(), [str(s).encode('utf-8') for s in heads],
        shared_op_indices, set(str(s).encode('utf-8') for s in shared_blobs),
        namescope.encode('utf-8'),
        set() if dont_share_blobs is None else dont_share_blobs,
        {} if blob_shapes is None else blob_shapes)

    log.info(
        "Memonger memory optimization took {} secs".format(time.time() -
                                                           start_time), )

    optim = caffe2_pb2.NetDef()
    optim.ParseFromString(optim_str)
    assert verify_graph_equality(net.Proto(), optim), \
        "Memonger graph is not equal to original."
    assert verify_inplace_blobs(net.Proto(), optim), \
        "Inplace assignments differ in memonger net."
    return optim


def deep_share_blobs(
        net,
        namescope,
        dont_share_blobs=None,
):
    external_input = list(set(net.Proto().external_input))

    input_shared_candidate = [
        namescope + s
        for s in ['__m0_shared', '__m1_shared', '__m2_shared', '__m3_shared']
    ]

    input_grad_candidate = []

    def is_fc_grad_blob(b):
        name = str(b)
        # Note: need to look at _{namescope} pattern as it matches
        # to handle the auto-split gradients
        return "fc" in name and (
            "_w_" in name or "_b_" in name) and name.endswith("_grad") and (
                name.startswith(namescope) or name.startswith("_" + namescope))

    for op_idx in range(len(net._net.op)):
        op = net._net.op[op_idx]
        for b in list(op.input) + list(op.output):
            if is_fc_grad_blob(b):
                input_grad_candidate.append(str(b))

    def is_new_blob(b):
        name = str(b)
        # Note: need to look at _{namescope} pattern as it matches
        # to handle the auto-split gradients
        return b not in external_input + input_shared_candidate + input_grad_candidate and (
            name.startswith(namescope) or name.startswith("_" + namescope))

    def is_shared_blob(b):
        name = str(b)
        # Note: need to look at _{namescope} pattern as it matches
        # to handle the auto-split gradients
        return (name.endswith('_shared')
                or b in input_shared_candidate + input_grad_candidate) and (
                    name.startswith(namescope)
                    or name.startswith("_" + namescope))

    log.warn("NOTE: Executing DEEP memonger to optimize gradient memory")

    # Collect ops that have something to do with gradients
    if namescope != "" and not namescope.endswith("/"):
        namescope += "/"

    shared_candidate = input_shared_candidate[:]
    shared_idx = len(shared_candidate)
    cnt_remap = 0

    for op_idx in range(len(net._net.op)):
        op = net._net.op[op_idx]
        for b in list(op.output):
            if str(b) in dont_share_blobs:
                continue

            if is_shared_blob(b):
                continue

            if is_new_blob(b):
                if len(shared_candidate) == 0:
                    shared_blob_name = namescope + '__m{}_shared'.format(
                        shared_idx)
                    shared_idx += 1
                else:
                    shared_blob_name = shared_candidate.pop(0)
                cnt_remap += 1

                for opop in net._net.op[op_idx:]:
                    for ii, bb in enumerate(opop.input):
                        if bb == b:
                            opop.input[ii] = shared_blob_name

                    for ii, bb in enumerate(opop.output):
                        if bb == b:
                            opop.output[ii] = shared_blob_name

        for b in list(op.input) + list(op.output):
            if is_shared_blob(b):
                last_used = True
                for opop in net._net.op[op_idx + 1:]:
                    for ii, bb in enumerate(opop.input):
                        if bb == b:
                            last_used = False
                if last_used:
                    shared_candidate.append(str(b))
                    shared_candidate = sorted(shared_candidate)
                    print(shared_candidate)

    log.warn("NOTE: remap {} blobs using {} shared blobs".format(
        cnt_remap, shared_idx))

    # inputs = []
    # for i in range(shared_idx):
    # s = namescope + '__m{}_shared'.format(i)
    # inputs.append(s)
    # net._net.external_input.extend(inputs)
    net._net.external_input.extend(input_shared_candidate)


def release_blobs_when_used(netproto, dont_free_blobs, selector_fun=None):
    '''
    Insert Free-ops after a blob has been used the last time, so that its
    memory can be reclaimed. Use this only with efficient caching memory
    managers (such as CUB, --caffe2_cuda_memory_pool=cub).

    Blobs used with Alias op won't be freed.

    @dont_free_blobs:  is a set of blobs that should not be freed
    @selector_fun:     optional lambda that return True if blob name
                       can be released. Use for easy special filtering, like
                       excluding blobs with "loss" in the name.

    Returns a new protobuffer. To use with a model, use:
        model.net._net = memonger.release_blobs_when_used(..)
    '''
    input_blobs = set()
    can_release = set()
    alias_blobs = set()
    netproto = copy.deepcopy(netproto)

    for op in netproto.op:
        if op.type == 'Alias':
            alias_blobs.add(op.input[0])
            continue
        for inp in op.input:
            input_blobs.add(inp)
        for outp in op.output:
            if outp not in input_blobs:
                if selector_fun is None or selector_fun(outp):
                    can_release.add(outp)

    # Remove such blobs that are not input at all and external outputs
    can_release = can_release - set(netproto.external_output)
    # can_release = can_release.intersection(input_blobs)
    can_release = can_release - dont_free_blobs
    can_release = can_release - alias_blobs

    ops = list(netproto.op)

    # .. then find last use of each can-release blob, and insert a Free op
    for j in reversed(range(0, len(netproto.op))):
        op = netproto.op[j]
        for inp in op.input:
            if inp in can_release:
                can_release.remove(inp)
                ops.insert(j + 1, core.CreateOperator("Free", [inp], [inp]))

    del netproto.op[:]
    netproto.op.extend(ops)
    return netproto


def deep_release_blobs_when_used(netproto, dont_free_blobs, selector_fun=None):
    '''
    Insert Free-ops after a blob has been used the last time, so that its
    memory can be reclaimed. Use this only with efficient caching memory
    managers (such as CUB, --caffe2_cuda_memory_pool=cub).

    Blobs used with Alias op won't be freed.

    @dont_free_blobs:  is a set of blobs that should not be freed
    @selector_fun:     optional lambda that return True if blob name
                       can be released. Use for easy special filtering, like
                       excluding blobs with "loss" in the name.

    Returns a new protobuffer. To use with a model, use:
        model.net._net = memonger.release_blobs_when_used(..)
    '''
    can_release = set()
    alias_blobs = set()
    netproto = copy.deepcopy(netproto)

    for op in netproto.op:
        if op.type == 'Alias':
            alias_blobs.add(op.input[0])
            continue
        for outp in op.output:
            if selector_fun is None or selector_fun(outp):
                can_release.add(outp)

    can_release = can_release - set(netproto.external_output)
    can_release = can_release - dont_free_blobs
    can_release = can_release - alias_blobs

    ops = list(netproto.op)

    # .. then find last use of each can-release blob, and insert a Free op
    for j in reversed(range(0, len(netproto.op))):
        op = netproto.op[j]
        for inp in op.input:
            if inp in can_release:
                can_release.remove(inp)
                ops.insert(j + 1, core.CreateOperator("Free", [inp], [inp]))

        for outp in op.output:
            if outp in can_release:
                can_release.remove(outp)
                ops.insert(j + 1, core.CreateOperator("Free", [outp], [outp]))

    assert len(can_release) == 0, can_release

    del netproto.op[:]
    netproto.op.extend(ops)
    return netproto


def verify_inplace_blobs(net_a, net_b):
    """
    Verifies that net_a and net_b have the same in-place blob assignments.
    Particularly, that memonger did not add an in-place assignment when that
    did not exist before.
    """

    def get_inplaces(op):
        out = list(op.output)
        inplaces = []
        for j, inp in enumerate(op.input):
            if inp in out:
                inplaces.append([j, out.index(inp)])
        return inplaces

    for op_a, op_b in zip(net_a.op, net_b.op):
        if op_a.type != op_b.type:
            return False
        if get_inplaces(op_a) != get_inplaces(op_b):
            return False
    return True


def verify_graph_equality(net_a, net_b):
    """
    Determines if the execution of two graphs are identical.
    That is, all inputs blobs are mapped to the same output blobs
    for each operator in their respective positions.

    This is meant to check the output of memonger with the original graph.
    It assumes that the nets have same external input and output.

    O(E) runtime + O(1) amortized cost to hash for python dict
    """

    def parent_list(ops):
        parent_list = [[] for _ in ops]
        edge_owner = {}
        for i, op in enumerate(ops):
            for blob in op.input:
                parent_id = edge_owner.get(blob)
                if parent_id is not None:
                    parent_list[i].append(parent_id)
            for blob in op.output:
                edge_owner[blob] = i

        return parent_list

    # Operator wise equality checks
    if (len(net_a.op) != len(net_b.op)):
        return False
    for op_a, op_b in zip(net_a.op, net_b.op):
        if (op_a.type != op_b.type or op_a.device_option != op_b.device_option
                or op_a.engine != op_b.engine):
            return False

    # Print debug info
    parent_list_a = parent_list(net_a.op)
    parent_list_b = parent_list(net_b.op)
    if parent_list_a != parent_list_b:
        j = 0
        for a, b in zip(parent_list_a, parent_list_b):
            if a != b:
                print("Difference {} vs {} \n {}".format(
                    j, net_a.op[j], net_b.op[j]))
                print("Parents: {} vs {}".format(a, b))

            j += 1

    # Net wise equality check
    return parent_list_a == parent_list_b
