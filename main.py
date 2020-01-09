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
from config.spm_config import spm_config as params
from loss.losses import spm_loss

import os
import datetime


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)
        # or set limited memory usage
        # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

    inputs = tf.keras.Input(shape=(params['height'], params['width'], 3),name='modelInput')
    outputs = SpmModel(inputs, num_joints=params['num_joints'], is_training=True)
    model = tf.keras.Model(inputs, outputs)

    if params['finetune'] is not None:
        model.load_weights(params['finetune'])
        print ('Successfully load pretrained model from ... {}'.format(params['finetune']))

    cur_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M')
    summary_writer = tf.summary.create_file_writer(os.path.join('./logs/spm', cur_time))

    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    train_dataset = get_dataset(num_gpus=1, mode='train')
    test_dataset  = get_dataset(num_gpus=1, mode='test')
    epochs = 150
    
    def lr_decay(epoch):
        if epoch < 90:
            return 1e-3
        elif epoch < 120:
            return 1e-4
        else:
            return 1e-5

    @tf.function
    def train_step(model, inputs):
        return model(inputs)

    def train_epoch():
        total_train_numbs = 1
        for epoch in range(epochs):
            for step, (img, center_map, kps_offset, kps_map_weight) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    center_pred, kps_offset_pred = train_step(model, img)
                    center_loss, kps_offset_loss = spm_loss(center_map, kps_offset, kps_map_weight, center_pred, kps_offset_pred)
                    center_loss /= params['batch_size']
                    kps_offset_loss /= params['batch_size'] 
                    train_batch_loss = center_loss+kps_offset_loss 

                grads = tape.gradient(train_batch_loss, model.trainable_variables)
                optimizer.learning_rate = lr_decay(epoch)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                tf.summary.scalar('loss', train_batch_loss, step=total_train_numbs*epoch+step)
                tf.summary.scalar('root_joint_loss', center_loss, step=total_train_numbs*epoch+step)
                tf.summary.scalar('offset_loss', kps_offset_loss, step=total_train_numbs*epoch+step)
                if step % 1 == 0:
                    gt_root_joints = center_map
                    pred_root_joints = center_pred
                    tf.summary.image('gt_root_joints', gt_root_joints, step=total_train_numbs*epoch+step, max_outputs=3)
                    tf.summary.image('pred_root_joints',pred_root_joints, step=total_train_numbs*epoch+step, max_outputs=3)
                    tf.summary.image('img', img, step=total_train_numbs*epoch+step, max_outputs=3)
                    print('for epoch {} step {}'.format(epoch, step))
                    print('....loss == {}, root joint loss == {}, body joint loss == {}'.format(train_batch_loss, center_loss, kps_offset_loss))

            if total_train_numbs == 1:
                total_train_numbs = step

            total_val_loss = 0.
            total_center_loss = 0.
            total_offset_loss = 0.
            for step, (img, center_map, kps_offset, kps_map_weight) in enumerate(test_dataset):
                center_pred, kps_offset_pred = train_step(model, img)
                center_loss, kps_offset_loss = spm_loss(center_map, kps_offset, kps_map_weight, center_pred, kps_offset_pred)
                total_center_loss += center_loss / params['batch_size']
                total_offset_loss += kps_offset_loss / params['batch_size']
                total_val_loss += (total_center_loss + total_offset_loss)
            print('......................................................................................\n, Epoch {}, ave center loss {:7f}, ave offset loss {:7f}'.format(epoch, total_center_loss / (step + 1), total_offset_loss / (step + 1)))

            model.save_weights('keras/{:02d}-{:.7f}.h5'.format(epoch+1, total_val_loss / (step + 1)))



    with summary_writer.as_default():
        train_epoch()

