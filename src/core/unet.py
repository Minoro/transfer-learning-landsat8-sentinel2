
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, Input, concatenate

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_height=256, input_width=256, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=10):
    input_img = Input(shape=(input_height,input_width, n_channels))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    model = tf.keras.Model(inputs=[input_img], outputs=[outputs])
    return model
    

def freeze_unet_layers(unet, mode):
    if mode == 'freeze_all':
        unet.trainable = False
        for layer in unet.layers:
            layer.trainable = False
    
    elif mode == 'freeze_encoder':
        for layer in unet.layers[:39]:
            layer.trainable = False

    elif mode == 'freeze_encoder_tune_bn':
        for layer in unet.layers[:39]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    elif mode == 'freeze_decoder':
        for layer in unet.layers[39:]:
            layer.trainable = False

    elif mode == 'freeze_decoder_tune_bn':
        for layer in unet.layers[39:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    elif mode == 'unfreeze':
        unet.trainable = True
        for layer in unet.layers:
            layer.trainable = True
            
    elif mode == 'tune_bn_encoder':
        for layer in unet.layers:
            layer.trainable = False

        for layer in unet.layers[:39]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    elif mode == 'tune_bn_decoder':
        for layer in unet.layers:
            layer.trainable = False

        for layer in unet.layers[39:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    elif mode == 'tune_bn':
        for layer in unet.layers:
            layer.trainable = False

        for layer in unet.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    else:
        raise 'Configuração não conhecida'

    return unet

# if __name__ == '__main__':

#     model = get_unet()

    # print(model.summary())