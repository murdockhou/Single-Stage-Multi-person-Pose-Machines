#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: decode_spm.py
@time: 2019/9/9 下午4:32
@desc:
'''
import numpy as np
import math
from utils.utils import point_nms
from config.spm_config import spm_config as params

class SpmDecoder():
    def __init__(self, factor_x, factor_y, outw, outh):
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.level = [[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11],
                      ]
        self.Z = math.sqrt(outw**2 + outh**2)
        # self.Z = 1
        self.outw = outw
        self.outh = outh
        # print ('decoder self.z', self.Z)
    def __call__(self, spm_label, score_thres=0.9, dis_thres=10):

        center_map = spm_label[0]
        kps_map = spm_label[1]

        keep_coors = point_nms(center_map, score_thres, dis_thres)
        centers = keep_coors[0]
        results = []

        for center in centers:
            single_person_joints = [0 for i in range(params['num_joints']*2)]
            root_joint = [int(x) for x in center]

            if root_joint[0] >= kps_map.shape[1] or root_joint[1] >= kps_map.shape[0] \
                    or root_joint[0] < 0 or root_joint[1] < 0:
                print ('find center point on wrong location')
                continue

            for single_path in self.level:
                start_joint = [root_joint[1], root_joint[0]]
                for i, index in enumerate(single_path):
                    offset = kps_map[root_joint[0], root_joint[1], 2*index:2*index+2]
                    if offset[0] == 0 or offset[1] == 0:
                        continue
                    joint = [start_joint[0]+offset[0]*self.Z, start_joint[1]+offset[1]*self.Z]
                    # print ('start joint {} -> end joint {}'.format(start_joint, joint))
                    single_person_joints[2*index:2*index+2] = joint
                    start_joint = joint

            for i in range(params['num_joints']):
                single_person_joints[2*i] *= self.factor_x
                single_person_joints[2*i+1] *= self.factor_y

            results.append({
                'center': [center[1] * self.factor_x, center[0] * self.factor_y, center[2]],
                'joints': single_person_joints
            })

        return results
