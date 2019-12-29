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

class SpmDecoder():
    def __init__(self, factor_x, factor_y, outw, outh):
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.level = [[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11],
                      [13, 12]]
        # self.Z = math.sqrt(outw*outw + outh*outh)
        self.Z = 1
        self.outw = outw
        self.outh = outh
        # print ('decoder self.z', self.Z)
    def __call__(self, spm_label, score_thres=0.9, dis_thres=5):

        center_map = spm_label[0]
        kps_map = spm_label[1]

        keep_coors = point_nms(center_map, score_thres, dis_thres)
        centers = keep_coors[0]
        # print (len(centers))
        joints = []
        ret_centers = []
        for center in centers:
            single_person_joints = [0 for i in range(14*2)]
            root_joint = [int(x) for x in center]

            if root_joint[0] >= kps_map.shape[1] or root_joint[1] >= kps_map.shape[0] \
                    or root_joint[0] < 0 or root_joint[1] < 0:
                print ('find center point on wrong location')
                continue

            for single_path in self.level:
                start_joint = [root_joint[1], root_joint[0]]
                for i, index in enumerate(single_path):
                    offset = kps_map[root_joint[0], root_joint[1], 2*index:2*index+2] * self.Z
                    # print (offset)
                    joint = [start_joint[0]+offset[0], start_joint[1]+offset[1]]
                    # print ('start joint {} -> end joint {}'.format(start_joint, joint))
                    single_person_joints[2*index:2*index+2] = joint
                    start_joint = joint

            ret_centers.append([center[1] * self.factor_x, center[0] * self.factor_y, center[2]])
            joints.append(single_person_joints)

        for single_person_joints in joints:
            for i in range(14):
                single_person_joints[2*i] *= self.factor_x
                single_person_joints[2*i+1] *= self.factor_y

        return joints, ret_centers
    # def __call__(self, spm_label, score_thres=0.9, dis_thres=5):
    #
    #     center_map = spm_label[0]
    #     kps_map = spm_label[1]
    #
    #     keep_coors = point_nms(center_map, score_thres, dis_thres)
    #     centers = keep_coors[0]
    #     # print (len(centers))
    #     joints = []
    #     ret_centers = []
    #     for center in centers:
    #         single_person_joints = [0 for i in range(14*2)]
    #         root_joint = [int(x) for x in center]
    #         ret_centers.append([center[1] * self.factor_x, center[0] * self.factor_y, center[2]])
    #         for single_path in self.level:
    #             # print (single_path)
    #             for i, index in enumerate(single_path):
    #                 # print (i, index)
    #                 if i == 0:
    #                     start_joint = [root_joint[1], root_joint[0]]
    #                 if start_joint[0] >= kps_map.shape[1] or start_joint[1] >= kps_map.shape[0] \
    #                         or start_joint[0] < 0 or start_joint[1] < 0:
    #                     break
    #                 offset = kps_map[start_joint[1], start_joint[0], 2*index:2*index+2] * self.Z
    #                 # print (offset)
    #                 joint = [start_joint[0]+offset[0], start_joint[1]+offset[1]]
    #                 # print ('start joint {} -> end joint {}'.format(start_joint, joint))
    #                 single_person_joints[2*index:2*index+2] = joint
    #                 start_joint = [int(x) for x in joint]
    #
    #         joints.append(single_person_joints)
    #
    #     for single_person_joints in joints:
    #         for i in range(14):
    #             single_person_joints[2*i] *= self.factor_x
    #             single_person_joints[2*i+1] *= self.factor_y
    #
    #     return joints, ret_centers