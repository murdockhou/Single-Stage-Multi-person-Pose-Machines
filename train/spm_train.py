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
from config.spm_config import spm_config as params

import os

@tf.function
def infer(model, inputs):

    preds = model(inputs)

    return preds

def train(model, optimizer, dataset, epochs, cur_time='8888-88-88-88'):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    if params['finetune'] is not None:
        manager = tf.train.CheckpointManager(ckpt, params['finetune'], max_to_keep=200)
        ckpt.restore(params['finetune'])
        print('successfully restore model from ... {}'.format(params['finetune']))
    else:
        manager = tf.train.CheckpointManager(ckpt, os.path.join(params['ckpt'], cur_time), max_to_keep=200)

    for epoch in range(epochs):
        for step, (img, center_map, kps_map, kps_map_weight) in enumerate(dataset):
            with tf.GradientTape() as tape:
                preds = infer(model, img)
                loss = spm_loss(center_map, kps_map, kps_map_weight, preds)
            grads = tape.gradient(loss[0], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            ckpt.step.assign_add(1)
            tf.summary.scalar('loss', loss[0], step=int(ckpt.step))
            tf.summary.scalar('root_joint_loss', loss[1], step=int(ckpt.step))
            tf.summary.scalar('offset_loss', loss[2], step=int(ckpt.step))
            if step % 100 == 0:
                gt_root_joints = center_map
                pred_root_joints = preds[0]
                tf.summary.image('gt_root_joints', gt_root_joints, step=int(ckpt.step), max_outputs=3)
                tf.summary.image('pred_root_joints',pred_root_joints, step=int(ckpt.step), max_outputs=3)
                tf.summary.image('img', img, step=int(ckpt.step), max_outputs=3)
                print('for epoch {} step {}'.format(epoch, step))
                print('....loss == {}, root joint loss == {}, body joint loss == {}'.format(loss[0], loss[1], loss[2]))
        save_path = manager.save()
        print('Saved ckpt for step {} : {}'.format(int(ckpt.step), save_path))