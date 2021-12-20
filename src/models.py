# %% CNN models definitions

from tensorflow import keras
import tensorflow.keras.layers as layers

from tensorflow.keras import Model
from tensorflow.keras import applications as default_models

# %%

def input_layer(single_input_shape=[256, 256, 1], num_inputs=1):
    if num_inputs is 1:
        return layers.Input(shape=single_input_shape)

    elif num_inputs > 1:
        return [layers.Input(shape=single_input_shape)
                for _ in range(num_inputs)]
    
    
def concatenate_block(input_tensor, **kwargs):
    return layers.concatenate(input_tensor)


def final_block_dense(input_tensor, pooling='avg', output_classes=2,
                      output_activation='softmax', dropout=0.3, 
                      kernel_regularizer=keras.regularizers.l2(0.01),
                      activity_regularizer = keras.regularizers.l1(0.01), 
                      intermediate_nodes=None, intermediate_activation='relu'):

    if pooling is None:
        x = layers.Flatten()(input_tensor)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(input_tensor)
    elif pooling == 'max':
        x = layers.MaxPooling2D()(input_tensor)

    if intermediate_nodes is not None:
        for nodes in intermediate_nodes:
            x = layers.Dense(nodes, activation=intermediate_activation, 
                             kernel_initializer='he_normal',
                             kernel_regularizer=kernel_regularizer,
                             activity_regularizer=activity_regularizer)(x)
    
    x = layers.Dropout(dropout)(x)
    output_layer = layers.Dense(output_classes, activation=output_activation, 
                                kernel_initializer='he_normal', 
                                kernel_regularizer=kernel_regularizer,
                                activity_regularizer=activity_regularizer)(x)

    return output_layer


def conv_encoder(input_tensor, filters=[32, 64, 64, 128, 128, 512, 512, 1024], dropout=0.2, activation='relu', residual_connection=False, scaling_input=None, pooling='Max',name='Conv_Encoder'):

    input_shape = input_tensor.shape
    x = input_tensor

    for idx, filt in enumerate(filters):

        name = name+'_EncoderLayer_'+str(idx)+'_'

        if idx > 0 and idx % 2 == 0:
            if pooling == 'Max':
                x = layers.MaxPool2D((2, 2), name=name+'MaxPool')(x)
            elif pooling == 'Avg':
                x = layers.AvgPool2D((2, 2), name=name+'AvgPool')(x)

        x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
        x = layers.Activation(activation, name=name+'Activation')(x)
        x = layers.Dropout(dropout)(x)
        x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    return x


def conv_encoder_atrous(input_tensor, filters=[32, 64, 64, 128, 128, 512, 512, 1024], activation='relu'):
    """-------------------Not Complete (Dilation Rate Logic)---------------------"""

    input_shape = input_tensor.shape
    x = input_tensor

    for idx, filt in enumerate(filters):

        name = 'EncoderLayer_'+str(idx)+'_'

        if idx > 0 and idx % 2 == 0:
            x = layers.Conv2D(filt, 3, dilation_rate=idx, padding='same', name=name+'DilatedConv2D', kernel_initializer='he_normal')(x)
            x = layers.Activation(activation, name=name+'Activation')(x)
            x = layers.BatchNormalization(name=name+'BatchNorm')(x)
        else:
            x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
            x = layers.Activation(activation, name=name+'Activation')(x)
            x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    return x


def conv_decoder(input_tensor, output_shape, intermediate_activation='relu', interpolation='bilinear', final_activation='relu'):

    input_shape = input_tensor.shape
    x = input_tensor
    filters = int(input_shape[-1])
    height = int(input_shape[1])

    count = 0

    while (height < output_shape[0]):

        tmp_filters = int(filters/2)

        name = 'DecoderLayer_'+str(count)+'_'

        x = keras.layers.Conv2D(tmp_filters, (3, 3),
                                activation=intermediate_activation,
                                padding='same', name=name+'Conv2D_A', kernel_initializer='he_normal')(x)
        x = keras.layers.Conv2D(tmp_filters, (3, 3),
                                activation=intermediate_activation,
                                padding='same', name=name+'Conv2D_B', kernel_initializer='he_normal')(x)
        x = keras.layers.UpSampling2D((2, 2), name=name+'UpSampling', interpolation=interpolation)(x)

        filters = int(x.shape[-1])
        height = int(x.shape[1])
        count += 1

    x = keras.layers.Conv2D(int(filters/2), (3, 3), activation=intermediate_activation,
                            padding='same', name='Final_Conv2D_A', kernel_initializer='he_normal')(x)
    x = keras.layers.Conv2D(int(output_shape[-1]), (3, 3), activation=final_activation,
                            padding='same', name='Final_Conv2D_B', kernel_initializer='he_normal')(x)

    return x


def cnn_autoencoder(input_tensor, output_shape, filters=[32, 64, 64, 128, 128, 512, 512, 1024], dilated=False, intermediate_activation='relu', interpolation='bilinear', final_activation='relu'):

    if dilated is True:
        enc = conv_encoder_atrous(input_tensor, filters=filters, activation=intermediate_activation)
    else:
        enc = conv_encoder(input_tensor, filters=filters, activation=intermediate_activation)

    dec = conv_decoder(enc, output_shape, intermediate_activation='relu', interpolation='bilinear', final_activation='sigmoid')

    return dec


def unet_upsampling(input_tensor, output_shape, filters=[32, 64, 64, 128, 128, 512, 512, 1024], bottleneck_filters=None, intermediate_activation='relu', interpolation='bilinear', final_activation='relu'):

    if bottleneck_filters is None:
        bottleneck_filters = filters[-2:]
        filters = filters[:-2]

    # ENCODER -------------------

    input_shape = input_tensor.shape
    x = input_tensor

    unet_connections = []

    for idx, filt in enumerate(filters):

        name = 'EncoderLayer_'+str(idx)+'_'

        if idx > 0 and idx % 2 == 0:
            unet_connections.append(x)
            x = layers.MaxPool2D((2, 2), name=name+'MaxPool')(x)

        x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
        x = layers.Activation(intermediate_activation, name=name+'Activation')(x)
        x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    # Bottleneck -------------------

    name = 'BottleNeck_'

    unet_connections.append(x)
    x = layers.MaxPool2D((2, 2), name=name+'MaxPool')(x)

    x = layers.Conv2D(bottleneck_filters[0], 3, padding='same', name=name+'Conv2D_A', kernel_initializer='he_normal')(x)
    x = layers.Activation(intermediate_activation, name=name+'Activation_A')(x)
    x = layers.BatchNormalization(name=name+'BatchNorm_A')(x)
    x = layers.Conv2D(bottleneck_filters[1], 3, padding='same', name=name+'Conv2D_B', kernel_initializer='he_normal')(x)
    x = layers.Activation(intermediate_activation, name=name+'Activation_B')(x)
    x = layers.BatchNormalization(name=name+'BatchNorm_B')(x)

    # DECODER -------------------

    filters.reverse()
    unet_connections.reverse()

    connection_num = 0

    for idx, filt in enumerate(filters):

        name = 'DecoderLayer_'+str(idx)+'_'

        if idx % 2 == 0:
            x = keras.layers.UpSampling2D((2, 2), name=name+'UpSampling', interpolation=interpolation)(x)
            x = keras.layers.concatenate([x, unet_connections[connection_num]], axis=-1, name=name+'Concatenate')
            connection_num += 1

        x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
        x = layers.Activation(intermediate_activation, name=name+'Activation')(x)
        x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    x = keras.layers.Conv2D(int(output_shape[-1]), (3, 3), activation=final_activation,
                            padding='same', name='Final_Conv2D', kernel_initializer='he_normal')(x)

    return x


def unet(input_tensor, output_shape, filters=[32, 64, 64, 128, 128, 512, 512, 1024], bottleneck_filters=None, intermediate_activation='relu', deconv_strides=2, final_activation='relu'):

    if bottleneck_filters is None:
        bottleneck_filters = filters[-2:]
        filters = filters[:-2]

    # ENCODER -------------------

    input_shape = input_tensor.shape
    x = input_tensor

    unet_connections = []

    for idx, filt in enumerate(filters):

        name = 'EncoderLayer_'+str(idx)+'_'

        if idx > 0 and idx % 2 == 0:
            unet_connections.append(x)
            x = layers.MaxPool2D((2, 2), name=name+'MaxPool')(x)

        x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
        x = layers.Activation(intermediate_activation, name=name+'Activation')(x)
        x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    # Bottleneck -------------------

    name = 'BottleNeck_'

    unet_connections.append(x)
    x = layers.MaxPool2D((2, 2), name=name+'MaxPool')(x)

    x = layers.Conv2D(bottleneck_filters[0], 3, padding='same', name=name+'Conv2D_A', kernel_initializer='he_normal')(x)
    x = layers.Activation(intermediate_activation, name=name+'Activation_A')(x)
    x = layers.BatchNormalization(name=name+'BatchNorm_A')(x)
    x = layers.Conv2D(bottleneck_filters[1], 3, padding='same', name=name+'Conv2D_B', kernel_initializer='he_normal')(x)
    x = layers.Activation(intermediate_activation, name=name+'Activation_B')(x)
    x = layers.BatchNormalization(name=name+'BatchNorm_B')(x)

    # DECODER -------------------

    filters.reverse()
    unet_connections.reverse()

    connection_num = 0

    for idx, filt in enumerate(filters):

        name = 'DecoderLayer_'+str(idx)+'_'

        if idx % 2 == 0:
            x = keras.layers.Conv2DTranspose(filt, 2, strides=deconv_strides, padding='valid', name=name+'DeConv')(x)
            x = keras.layers.concatenate([x, unet_connections[connection_num]], axis=-1, name=name+'Concatenate')
            x = layers.Activation(intermediate_activation, name=name+'Activation')(x)
            x = layers.BatchNormalization(name=name+'BatchNorm')(x)
            connection_num += 1
        else:
            x = layers.Conv2D(filt, 3, padding='same', name=name+'Conv2D', kernel_initializer='he_normal')(x)
            x = layers.Activation(intermediate_activation, name=name+'Activation')(x)
            x = layers.BatchNormalization(name=name+'BatchNorm')(x)

    x = keras.layers.Conv2D(int(output_shape[-1]), (3, 3), activation=final_activation,
                            padding='same', name='Final_Conv2D', kernel_initializer='he_normal')(x)

    return x


def resnet50(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.resnet50.ResNet50(**params)
    return model.output


def resnet_v2(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.resnet_common.ResNet50V2(**params)
    return model.output


def inception_resnetv2(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.inception_resnet_v2.InceptionResNetV2(**params)
    return model.output


def vgg_16(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.vgg16.VGG16(**params)
    return model.output


def nasnet(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.nasnet.NASNet(**params)
    return model.output

def densenet169(input_tensor, **kwargs):
    default_params = {
        'input_tensor': input_tensor,
        'weights': None,
        'include_top': False,
        'pooling': None,
    }
    params = {**default_params, **kwargs}
    model = default_models.DenseNet169(**params)
    return model.output
