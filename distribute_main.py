import tensorflow as tf
from dataset.dataset import get_dataset
from nets.spm_model import SpmModel
from config.center_config import center_config

import os
import datetime

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    gpu_ids = [0, 1]
    devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    checkpoint_dir = './models'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    epochs = 5

    with strategy.scope():
        inputs = tf.keras.Input(shape=(center_config['height'], center_config['width'], 3), name='modelInput')
        outputs = SpmModel(inputs, num_joints=center_config['joints'], is_training=True)
        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.optimizers.Adam(learning_rate=3e-4)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if center_config['finetune'] is not None:
            checkpoint.restore(center_config['finetune'])
            print('Successfully restore model from {}'.format(center_config['finetune']))

    with strategy.scope():
        dataset = get_dataset(len(gpu_ids))
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        print(dist_dataset.__dict__['_cloned_datasets'])


    def SmoothL1Loss(label, pred):
        t = tf.abs(label - pred)
        return tf.reduce_mean(
            tf.where(
                t <= 1, 0.5*t*t, 0.5*(t-1)
            )
        )

    def spm_loss(gt_root_joint, gt_joint_offset, gt_joint_offset_weight, preds):
        root_weight = 10
        joint_weight = 1

        # gt_root_joint = label[..., 0:1]
        # gt_joint_offset = label[..., 1:2 * 14 + 1]
        # gt_joint_offset_weight = label[..., 2 * 14 + 1:]

        pred_root_joint = preds[0]
        pred_joint_offset = preds[1]

        # root_joint_loss = tf.reduce_sum(tf.nn.l2_loss(pred_root_joint - gt_root_joint))
        root_joint_loss = tf.reduce_mean(tf.keras.losses.MSE(gt_root_joint, pred_root_joint))

        # huber loss 就是 smooth l1 loss，和pytorch中的torch.nn.SmoothL1Loss()结果一致
        # huber_loss = tf.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        # pred_joint_loss = tf.reduce_sum(huber_loss(gt_joint_offset * gt_joint_offset_weight,
        #                                            pred_joint_offset * gt_joint_offset_weight))

        pred_joint_loss = SmoothL1Loss(gt_joint_offset * gt_joint_offset_weight, pred_joint_offset * gt_joint_offset_weight)

        return root_weight * root_joint_loss + joint_weight * pred_joint_loss


    with strategy.scope():
        def train_step(inputs):
            img, center_map, kps_map, kps_map_weight = inputs
            with tf.GradientTape() as tape:
                preds = model(img)
                loss = spm_loss(center_map, kps_map, kps_map_weight, preds) * (
                            1.0 / (center_config['batch_size'] * len(gpu_ids)))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss


        @tf.function
        def distribute_train_step(inputs):
            per_replica_loss = strategy.experimental_run_v2(train_step, args=(inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


        for epoch in range(epochs):
            total_loss = 0.0
            train_batchs = 0.0

            for x in dist_dataset:
                total_loss += distribute_train_step(x)
                train_batchs += 1

            template = ('Epoch: {}, Train Steps: {}, Train Ave Loss: {}')
            print(template.format(epoch, train_batchs, total_loss / train_batchs))
            print('Model saved in {} '.format(checkpoint.save(checkpoint_prefix)))