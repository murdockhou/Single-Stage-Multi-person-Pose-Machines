#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: main.py
@time: 2019/7/25 下午3:07
@desc:
'''

import tensorflow as tf
from dataset.dataset import get_dataset
from nets.spm_model import SpmModel
from config.center_config import center_config
from train.spm_train import train

import os
import datetime


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    strategy = tf.distribute.MirroredStrategy(devices=['/gpu:5', '/gpu:6'])
    with strategy.scope():
        inputs = tf.keras.Input(shape=(center_config['height'], center_config['width'], 3),name='modelInput')
        outputs = SpmModel(inputs, num_joints=center_config['joints'], is_training=True)
        model = tf.keras.Model(inputs, outputs)

    cur_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M')

    optimizer = tf.optimizers.Adam(learning_rate=3e-4)
    dataset = get_dataset(2)
    epochs = 200
    summary_writer = tf.summary.create_file_writer(os.path.join('./logs/spm', cur_time))
    with summary_writer.as_default():
        train(model, optimizer, dataset, epochs, cur_time)
