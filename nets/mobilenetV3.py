import tensorflow as tf

def hardSwish(x):
    return tf.keras.layers.Multiply()([x, tf.keras.layers.ReLU(6)(x+3) / 6.])

def MBConvBlock(inputs, out_size, exp_size, use_se, activation, kernel_size, strides, training):

    y = inputs
    # pointwise
    x = tf.keras.layers.Conv2D(exp_size, kernel_size=1, strides=(1,1), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = activation(x)
    # depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=(strides, strides), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = activation(x)
    # se
    if use_se:
        x = SEBlock(x)
    # pointwise
    x = tf.keras.layers.Conv2D(out_size, kernel_size=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = activation(x)
    # residual
    if strides == 1 and y.shape[-1] == x.shape[-1]:
        x = y + x

    return x

def SEBlock(inputs):
    input_channels = inputs.shape[-1]
    y = inputs
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(input_channels//4, activation='relu')(x)
    x = tf.keras.layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = tf.keras.layers.Reshape([1, 1, input_channels])(x)
    return tf.keras.layers.Multiply()([x, y])
    
def up_sample(inputs, times, training=True):
    net = inputs
    while times > 0:
        x = tf.keras.layers.UpSampling2D()(net)
        x = tf.keras.layers.BatchNormalization()(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        times -= 1
        net = x
    return net

RE = tf.keras.layers.ReLU()
HS = hardSwish

# kernel size, exp size, out size, se block, RE/HS, stride
large_params = [
    [3, 16, 16, False, RE, 1],
    [3, 64, 24, False, RE, 2],
    [3, 72, 24, False, RE, 1],
    [5, 72, 40, True,  RE, 2],
    [5, 120, 40, True, RE, 1],
    [5, 120, 40, True, RE, 1],
    [3, 240, 80, False, HS, 2],
    [3, 200, 80, False, HS, 1],
    [3, 184, 80, False, HS, 1],
    [3, 184, 80, False, HS, 1],
    [3, 480, 112, True, HS, 1],
    [3, 672, 112, True, HS, 1],
    [5, 672, 160, True, HS, 2],
    [5, 960, 160, True, HS, 1],
    [5, 960, 160, True, HS, 1],
]
large_params_layers = [2, 5, 11, 14]

small_params = [
    [3, 16, 16, True, RE, 2],
    [3, 72, 24, False, RE, 2],
    [3, 88, 24, False, RE, 1],
    [5, 96, 40, True, HS, 2],
    [5, 240, 40, True, HS, 1],
    [5, 240, 40, True, HS, 1],
    [5, 120, 48, True, HS, 1],
    [5, 144, 48, True, HS, 1],
    [5, 288, 96, True, HS, 2],
    [5, 576, 96, True, HS, 1],
    [5, 576, 96, True, HS, 1],
]
small_params_layers = [0, 2, 7, 10]

def MobileNetV3Large(inputs, training=True):

    features = []

    x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = hardSwish(x)

    for layer, param in enumerate(large_params):
        x = MBConvBlock(x, out_size=param[2], exp_size=param[1], use_se=param[3], activation=param[4], kernel_size=param[0],
                        strides=param[5], training=training)
        if layer in large_params_layers:
            features.append(x)

    # x = tf.keras.layers.Conv2D(960, 1)(x)
    # x = tf.keras.layers.BatchNormalization()(x, training=training)
    # x = hardSwish(x)

    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Dense(1280)(x)
    # x = tf.keras.layers.Lambda(hardSwish)(x)
    # x = tf.keras.layers.Dense(1000)(x)

    # return x, features


    times= [0, 1, 2, 3]
    backbone = []
    for i, net in enumerate(features):
        net = up_sample(net, times[i], training=training)
        backbone.append(net)

    backbone = tf.keras.layers.Concatenate()(backbone)
    backbone = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(backbone)

    return backbone    

def MobileNetV3Small(inputs, training=True):

    features = []

    x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=(2,2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = hardSwish(x)

    for layer, param in enumerate(small_params):
        x = MBConvBlock(x, out_size=param[2], exp_size=param[1], use_se=param[3], activation=param[4], kernel_size=param[0],
                        strides=param[5], training=training)
        if layer in small_params_layers:
            features.append(x)

    x = tf.keras.layers.Conv2D(576, 1)(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = hardSwish(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1280)(x)
    x = tf.keras.layers.Lambda(hardSwish)(x)
    x = tf.keras.layers.Dense(1000)(x)

    return x, features



if __name__ == '__main__':
    import os
    import time
    from nets.single_posenet import infer

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    inputs = tf.keras.Input(shape=(224,224,3))
    outputs = MobileNetV3Small(inputs, training=True)
    model = tf.keras.models.Model(inputs, outputs)
    # tf.keras.utils.plot_model(model, 'mobilenetV3.png', show_shapes=True)
    model.summary()
    #
    total = 0.0
    a = tf.random.normal([1, 224, 224, 3])
    for i in range(110):
        s = time.time()
        infer(model, a)
        e = time.time()
        print (e-s)
        if i > 9:
            total += (e-s)

    print ('Avg 100 times, infer time == ', total/100)
    model.load_weights('model_trans/model/mobilenetv3.h5')
    # print (model.outputs[0].op.name)