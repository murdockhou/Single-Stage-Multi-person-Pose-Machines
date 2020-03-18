import tensorflow as tf
from dataset.dataset import get_dataset
from nets.spm_model import SpmModel
from config.spm_config import spm_config as params

import os
import datetime

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6, 7'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    gpu_ids = [0, 1, 2]
    devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    #checkpoint_dir = './models'
    #checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    epochs = 150

    def step_lr(epoch):
        if epoch < 90:
            return 1e-3
        elif epoch < 120:
            return 1e-4
        else:
            return 1e-5

    with strategy.scope():
        inputs = tf.keras.Input(shape=(params['height'], params['width'], 3), name='modelInput')
        outputs = SpmModel(inputs, num_joints=params['num_joints'], is_training=True)
        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.optimizers.Adam(learning_rate=3e-4)
        if params['finetune'] is not None:
            model.load_weights(params['finetune'])
            print('Successfully restore model from {}'.format(params['finetune']))

    with strategy.scope():
        dataset = get_dataset(len(gpu_ids))
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        #print(dist_dataset.__dict__['_cloned_datasets'])


    with strategy.scope():
        def SmoothL1Loss(label, pred, weight):
            t = tf.abs(label * weight - pred * weight)

            return tf.reduce_sum(
                tf.where(
                    t <= 1, 0.5 * t * t, 0.5 * (t - 1)
                )
            )

        def L2Loss(label, pred, weight=None):
            if weight is None:
                weight = 1. 
            return tf.reduce_sum(tf.nn.l2_loss(label*weight - pred*weight))

        def comput_loss(center_map, center_mask, kps_map, kps_map_weight, preds):
            kps_loss = SmoothL1Loss(kps_map, preds[1], kps_map_weight)
            root_loss = L2Loss(center_map, preds[0], weight=center_mask)
            per_example_loss =0.1* kps_loss + root_loss
            return per_example_loss
            #return tf.nn.compute_average_loss(per_example_loss, global_batch_size=params['batch_size']*len(gpu_ids))


    with strategy.scope():
        def train_step(inputs):
            img, center_map, center_mask, kps_map, kps_map_weight = inputs
            with tf.GradientTape() as tape:
                preds = model(img)
                loss = comput_loss(center_map, center_mask, kps_map, kps_map_weight, preds)
                loss = loss * 1.0 / (params['batch_size'] * len (gpu_ids))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss


        @tf.function
        def distribute_train_step(dataset_inputs):
            per_replica_loss = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)


        for epoch in range(epochs):
            total_loss = 0.0
            train_batchs = 0.0
            optimizer.learning_rate = step_lr(epoch)

            for x in dist_dataset:
                train_batch_loss = distribute_train_step(x)
                total_loss += train_batch_loss
                train_batchs += 1
               
            #checkpoint.save(checkpoint_prefix)
            model.save_weights('keras/'+'{:03d}.hdf5'.format(epoch+5))
            template = ('Epoch: {}, Train Steps: {}, Train Ave Loss: {}')
            print(template.format(epoch, train_batchs, total_loss / train_batchs))
