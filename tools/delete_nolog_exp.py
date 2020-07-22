#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import shutil

if __name__ == '__main__':

    exp_path = './experiments'
    log_path = './experiments/_logs'

    log_names = []
    for root, dirs, files, in os.walk(log_path):
        if './experiments/_logs' in root:
            pass
        else:
            continue
        print(root)
        for f in files:
            if f.endswith('.log'):
                print('Finding log: ', f)
                log_names.append(f)
            if f.endswith('.png'):
                print('Finding draw: ', f)

        for d in dirs:
            pass
            # print(d)

    log_ids = []
    for name in log_names:
        log_ids.append(name.split(' ')[0])

    print(log_ids)

    cnt_k = 0
    cnt_d = 0
    for root, dirs, files, in os.walk(exp_path):
        if root == './experiments':
            pass
        else:
            continue
        print(root)
        for d in sorted(dirs):
            if '_logs' in d:
                continue
            if d in log_ids:
                print(cnt_k, '******** Keeping: ', d)
                cnt_k += 1
            else:
                print(cnt_d, 'XXXXXXXX Deleting: ', d)
                cnt_d += 1

                p_d = os.path.join(root, d)
                shutil.rmtree(p_d)

    print('total log ids: ', len(log_ids))
    print('total deleteds: ', cnt_d)
    print('total keepeds: ', cnt_k)
