#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm.py
@time: 2019/9/9 下午2:54
@desc:
'''
import numpy as np
import math
from utils.utils import draw_gaussian, clip, draw_ttfnet_gaussian

class SingleStageLabel():
    def __init__(self, height, width, centers, sigmas, kps):
        self.centers = centers
        self.sigmas = sigmas
        self.kps = kps
        self.height = height
        self.width = width
        # self.Z = math.sqrt(height*height+width*width)
        self.Z = 1

        # print ('encoder: self.Z', self.Z)

        self.center_map = np.zeros(shape=(height, width, 1), dtype=np.float32)
        self.kps_map = np.zeros(shape=(height, width, 14*2), dtype=np.float32)
        self.kps_count = np.zeros(shape=(height, width, 14*2), dtype=np.uint)
        self.kps_map_weight = np.zeros(shape=(height, width, 14*2), dtype=np.float32)

        # hierarchical SPR, 0->1->2, 3->4->5, 6->7->8, 9->10->11, 13->12
        self.level = [[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11],
                      [13, 12]]

        # self.body_level = {
        #     14:[0, 3, 6, 9, 13],
        #     0:[1],
        #     1:[2],
        #     3:[4],
        #     4:[5],
        #     6:[7],
        #     7:[8],
        #     9:[10],
        #     10:[11],
        #     13:[12]
        # }
        # # root joints -> [0, 3, 6, 9, 13]
        # self.root_joints_reg_map = np.zeros(shape=(height, width, 5*2), dtype=np.float32)
        # # joints [0->1, 1->2, 3->4, 4->5, 6->7, 7->8, 9->10, 10->11, 13->12]
        # self.body_jonts_reg_map = np.zeros(shape=(9, height, width, 2), dtype=np.float32)
        #
        # # 前5个是root joint对于[0, 3, 6, 9, 13]的offset，后面依次是：[0->1, 1->2, 3->4, 4->5, 6->7, 7->8, 9->10, 10->11, 13->12]
        # self.reg_map = np.zeros(shape=(height, width, 14, 2), dtype=np.float32)
        #
        # # 对于root joint，只生成位于 0 3 6 9 13的offset
        # self.reg_map = np.zeros(shape=(height, width, 14*2), dtype=np.float32)


    def __call__(self):

        for i, center in enumerate(self.centers):
            sigma = self.sigmas[i]
            kps = self.kps[i]
            if center[0] == 0 and center[1] == 0:
                continue
            # self.center_map[..., 0] = draw_gaussian(self.center_map[...,0], center, sigma, mask=None)
            self.center_map[..., 0] = draw_ttfnet_gaussian(self.center_map[...,0], center, sigma[0], sigma[1])
            self.body_joint_displacement(center, kps, sigma)

        # print (np.where(self.kps_count > 2))
        self.kps_count[self.kps_count == 0] += 1
        self.kps_map = np.divide(self.kps_map, self.kps_count)

        return np.concatenate([self.center_map, self.kps_map, self.kps_map_weight], axis=-1)

    def body_joint_displacement(self, center, kps, sigma):
        # taux = sigma[0]
        # tauy = sigma[1]
        taux = 2
        tauy = 2

        for single_path in self.level:
            # print ('encoder single path: ', single_path)
            for i, index in enumerate(single_path):
                # print ('i {} : index {}'.format(i, index))
                if i == 0:
                    start_joint = center
                end_joint = kps[3*index:3*index+3]
                if start_joint[0] == 0 or start_joint[1] == 0:
                    continue
                if end_joint[0] == 0 or end_joint[1] == 0:
                    continue
                self.create_dense_displacement_map(index, start_joint, end_joint, taux, tauy)
                start_joint = end_joint

    def create_dense_displacement_map(self, index, start_joint, end_joint, sigmax=2, sigmay=2):

        # print('start joint {} -> end joint {}'.format(start_joint, end_joint))
        center_x, center_y = int(start_joint[0]), int(start_joint[1])
        th = 4.6052
        delta = np.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigmax + 0.5))
        y0 = int(max(0, center_y - delta * sigmay + 0.5))

        x1 = int(min(self.width, center_x + delta * sigmax + 0.5))
        y1 = int(min(self.height, center_y + delta * sigmay + 0.5))

        # x0 = int(clip(start_joint[0], 0, self.width))
        # y0 = int(clip(start_joint[1], 0, self.height))
        # x1 = int(clip(x0+taux, 0, self.width))
        # y1 = int(clip(y0+tauy, 0, self.height))
        # print (x0,x1, y0,y1)
        for x in range(x0, x1):
            for y in range(y0, y1):
                x_offset = (end_joint[0] - x) / self.Z
                y_offset = (end_joint[1] - y) / self.Z
                # print (x_offset, y_offset)
                self.kps_map[y, x, 2*index] += x_offset
                self.kps_map[y, x, 2*index+1] += y_offset
                self.kps_map_weight[y, x, 2*index:2*index+2] = 1.
                if end_joint[0] != x or end_joint[1] != y:
                    self.kps_count[y, x, 2*index:2*index+2] += 1

