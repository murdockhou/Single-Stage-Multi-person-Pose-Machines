import tensorflow as tf
from dataset.dataset import  get_dataset
from nets.spm_model import SpmModel
from config.spm_config import spm_config as params

import os

if __name__ =='__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    print('Visible devices : ', visible_gpus)

    gpu_ids = [0, 1]
    devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
    strategy = tf.distribute.MirroredStrategy(devices=devices)


    # def loss func
    def SmoothL1Loss(label, pred):

        weight = label[..., params['joints'] * 2:]
        label = label[..., :params['joints'] * 2]

        t = tf.abs(label * weight - pred * weight)

        return tf.reduce_mean(
            tf.where(
                t <= 1, 0.5 * t * t, 0.5 * (t - 1)
            )
        )


    def MSELoss(label, pred):
        return tf.reduce_mean(tf.keras.losses.MSE(label, pred))


    with strategy.scope():
        # define model
        inputs = tf.keras.Input(shape=(params['height'], params['width'], 3), name='modelInput')
        outputs = SpmModel(inputs, num_joints=params['joints'], is_training=True)
        model = tf.keras.Model(inputs, outputs)


        model.compile(loss={'root_joints_conv1x1': MSELoss, 'reg_map_conv1x1': SmoothL1Loss},
                      loss_weights={'root_joints_conv1x1': 1, 'reg_map_conv1x1': 1},
                      optimizer=tf.keras.optimizers.Adam(1e-4))

        if params['finetune'] is not None:
            model.load_weights(params['finetune'])


    # define dataset
    train_dataset = get_dataset(num_gpus=len(gpu_ids), mode='train')
    test_dataset  = get_dataset(num_gpus=len(gpu_ids), mode='val')
    def generator(dataset):
        for input, output1, output2 in dataset:
            yield input, {'root_joints_conv1x1':output1, 'reg_map_conv1x1':output2}


    def step_lr(epoch):
        if epoch < 20:
            return 1e-3
        elif epoch < 50:
            return 1e-4
        else:
            return 1e-5

    # def callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='keras/logs', write_graph=True, update_freq=100),
        tf.keras.callbacks.ModelCheckpoint(filepath='keras/{epoch:02d}-{val_loss:.7f}.hdf5', monitor='val_loss', verbose=1, save_weights_only=True),
        # tf.keras.callbacks.LearningRateScheduler(schedule=step_lr, verbose=1)
    ]

    # start training
    # model.fit_generator(generator(train_dataset), steps_per_epoch=10000, epochs=5, callbacks=callbacks,
    #                     validation_data=generator(test_dataset), validation_steps=10)
    model.fit(generator(train_dataset), steps_per_epoch=210000//(len(gpu_ids)*params['batch_size']), epochs=80, callbacks=callbacks,
                        validation_data=generator(test_dataset), validation_steps=30000//(len(gpu_ids)*params['batch_size']))



























    # def SmoothL1Loss(label, pred):
    #     t = tf.abs(label - pred)
    #     return tf.reduce_mean(tf.where(t <= 1, 0.5 * t * t, 0.5 * (t - 1)))
    #
    #
    # def root_loss(gt_root_joint, pred_root_joint):
    #     root_joint_loss = tf.reduce_mean(tf.keras.losses.MSE(gt_root_joint, pred_root_joint))
    #     return root_joint_loss
    #
    # inputs = tf.keras.Input(shape=(center_config['height'], center_config['width'], 3), name='modelInput')
    # outputs = SpmModel(inputs, num_joints=center_config['joints'], is_training=True)
    # model = tf.keras.Model(inputs, outputs)
    #
    # model.compile(loss={'root_joints_conv1x1': root_loss,
    #                     'reg_map_conv1x1': SmoothL1Loss},
    #               loss_weights={'root_joints_conv1x1': 10,
    #                             'reg_map_conv1x1': 1},
    #               optimizer=tf.keras.optimizers.Adam())
    #
    # dataset = get_dataset()
    #
    # model.fit(dataset, epochs=2)
