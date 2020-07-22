#!/usr/bin/env python2
"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import numpy as np

from caffe2.python import utils as c2_py_utils
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.utils.logging import log_json_stats
from detectron.utils.logging import SmoothedValue
from detectron.utils.timer import Timer
import detectron.utils.net_wsl as nu


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, model):
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = int(1280 / cfg.NUM_GPUS)
        # Output logging period in SGD iterations
        self.LOG_PERIOD = int(1280 / cfg.NUM_GPUS)
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(self.WIN_SZ)
            for key in model.losses + model.metrics
        }
        self.losses_and_metrics = {
            key: 0
            for key in model.losses + model.metrics
        }
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        self.smoothed_mb_qsize = SmoothedValue(self.WIN_SZ)
        self.iter_total_loss = np.nan
        self.iter_timer = Timer()
        self.model = model

        self.mem = dict()
        self.mem = None

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self):
        """Update tracked iteration statistics."""
        for k in self.losses_and_metrics.keys():
            self.losses_and_metrics[k] = nu.average_multi_gpu_blob(k)
        for k, v in self.smoothed_losses_and_metrics.items():
            v.AddValue(self.losses_and_metrics[k])
        self.iter_total_loss = np.sum(
            np.array([self.losses_and_metrics[k] for k in self.model.losses]))
        self.smoothed_total_loss.AddValue(self.iter_total_loss)
        self.smoothed_mb_qsize.AddValue(
            self.model.roi_data_loader._minibatch_queue.qsize())

        if self.mem is not None:
            self.GetMem()

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0
                or cur_iter == cfg.SOLVER.MAX_ITER - 1):
            stats = self.GetStats(cur_iter, lr)
            log_json_stats(stats)

        if self.mem is not None:
            mem_sorted = sorted(self.mem.items(), key=lambda d: d[1])
            print(mem_sorted)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            cfg.SOLVER.MAX_ITER - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        mem_stats = c2_py_utils.GetGPUMemoryUsageStats()
        mem_usage = np.max(mem_stats['max_by_gpu'][:cfg.NUM_GPUS])
        stats = dict(
            iter=cur_iter,
            lr=float(lr),
            time=self.iter_timer.average_time,
            loss=self.smoothed_total_loss.GetAverageValue(),
            eta=eta,
            mb_qsize=int(np.round(self.smoothed_mb_qsize.GetAverageValue())),
            mem=int(np.ceil(mem_usage / 1024 / 1024)))
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = v.GetAverageValue()
        return stats

    def is_grad(self, b):
        name = str(b)

        return name.endswith("_grad")

    def is_shared(self, b):
        name = str(b)

        return name.endswith("_shared")

    def GetMem(self):
        for op_idx in range(len(self.model.net._net.op)):
            op = self.model.net._net.op[op_idx]
            for b in list(op.output):
                if self.is_grad(b):
                    pass
                elif self.is_shared(b):
                    pass
                else:
                    continue

                blob = workspace.FetchBlob(str(b))
                if b not in self.mem.keys():
                    self.mem[str(b)] = 0
                self.mem[str(b)] = max(self.mem[str(b)], blob.size)
