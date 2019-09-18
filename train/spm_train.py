#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm_train.py
@time: 2019/9/10 上午11:46
@desc:
'''

import tensorflow as tf
from loss.losses import spm_loss
from config.center_config import center_config

import os

def train(model, optimizer, dataset, epochs, cur_time='8888-88-88-88'):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    if center_config['finetune'] is not None:
        manager = tf.train.CheckpointManager(ckpt, center_config['finetune'], max_to_keep=200)
        ckpt.restore(manager.checkpoints[-1])
    else:
        manager = tf.train.CheckpointManager(ckpt, os.path.join(center_config['ckpt'], cur_time), max_to_keep=200)

    for epoch in range(epochs):
        for step, (img, label) in enumerate(dataset):
            with tf.GradientTape() as tape:
                preds = model(img)
                loss = spm_loss(label, preds)
            grads = tape.gradient(loss[0], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ckpt.step.assign_add(1)
            tf.summary.scalar('loss', loss[0], step=int(ckpt.step))
            tf.summary.scalar('root_joint_loss', loss[1], step=int(ckpt.step))
            tf.summary.scalar('offset_loss', loss[2], step=int(ckpt.step))
            if int(ckpt.step) % 10 == 0:
                gt_root_joints = label[..., 0:1]
                pred_root_joints = preds[0]
                tf.summary.image('gt_root_joints', gt_root_joints, step=int(ckpt.step), max_outputs=3)
                tf.summary.image('pred_root_joints',pred_root_joints, step=int(ckpt.step), max_outputs=3)
                tf.summary.image('img', img, step=int(ckpt.step), max_outputs=3)
            if int(ckpt.step) % 20 == 0:
                print('for epoch {} step {}'.format(epoch, int(ckpt.step)))
                print('....loss == {}, root joint loss == {}, body joint loss == {}'.format(loss[0], loss[1], loss[2]))
            if int(ckpt.step) % 5000 == 0:
                save_path = manager.save()
                print('Saved ckpt for step {} : {}'.format(int(ckpt.step), save_path))