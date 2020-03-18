#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm_model.py
@time: 2019/9/10 上午10:47
@desc:
'''

import tensorflow as tf
# from nets.hrnet import HRNet as BackBone
from nets.mobilenetV3 import MobileNetV3Large as BackBone

def SpmModel(inputs, num_joints, is_training = True):

    body = BackBone(inputs, training=is_training)

    body = head_net(body, 512, name='body_1', training=is_training)
    body = head_net(body, 512, name='body_2', training=is_training)
    body = head_net(body, 512, name='body_3', training=is_training)
    rootJoints = head_net(body, 1, name='center', bn=False)
    displacement = head_net(body, 2*num_joints, name='center_kps_offset', bn=False, activation=tf.nn.tanh)

    return rootJoints, displacement

def head_net(inputs, output_c, name='', bn=True, training=True, activation=None):
    out = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=name + '_conv3x3')(inputs)
    if bn:
        out = tf.keras.layers.BatchNormalization()(out, training=training)

    out = tf.keras.layers.ReLU(name=name + '_relu')(out)
    out = tf.keras.layers.Conv2D(filters=output_c, kernel_size=1,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=name + '_conv1x1', activation=activation)(out)
    return out