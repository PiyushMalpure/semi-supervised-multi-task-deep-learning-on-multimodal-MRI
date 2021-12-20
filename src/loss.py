import os
import numpy as np
import nibabel as nib
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils.vis_utils import plot_model

"""------------------------------------------------------------------------------------------------"""


def weighted_entropy_loss(y_true, y_pred, weight={'0': 0.11, '1': 0.89}):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return -K.mean(weight['1']*y_true*K.log(y_pred) + weight['0']*(1-y_true)*K.log(1-y_pred))


def get_crossentropy(weight):

    def wt_crossentropy(y_true, y_pred):
        return weighted_entropy_loss(y_true, y_pred, weight)

    return wt_crossentropy


"""------------------------------------------------------------------------------------------------"""


def dice_score(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f)+K.epsilon())


def dice_loss(y_true, y_pred):
    return 1 - dice_score(y_true, y_pred)


"""------------------------------------------------------------------------------------------------"""


def tversky_loss(y_true, y_pred, weight={'alpha': 0.3, 'beta': 0.7}, smooth=K.epsilon()):

    #alpha = 0.3
    #beta = 0.7

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    tp = K.sum(y_true * y_pred)
    fp_and_fn = weight['alpha'] * K.sum(y_pred * (1 - y_true)) + weight['beta'] * K.sum((1 - y_pred) * y_true)
    return -(tp + smooth) / ((tp + smooth) + fp_and_fn)


def get_tversky(weight):

    def wt_tversky(y_true, y_pred):
        return tversky_loss(y_true, y_pred, weight)

    return wt_tversky


"""------------------------------------------------------------------------------------------------"""


def focal_loss(y_true, y_pred, weight={'alpha': 0.3, 'gamma': 2}):
    #alpha = 0.3
    #gamma = 2

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(weight['alpha'] * K.pow(1. - pt_1, weight['gamma']) * K.log(pt_1))-K.sum((1-weight['alpha']) * K.pow(pt_0, weight['gamma']) * K.log(1. - pt_0))


def get_focal(weight):

    def wt_focal(y_true, y_pred):
        return focal_loss(y_true, y_pred, weight)

    return wt_focal


"""------------------------------------------------------------------------------------------------"""


def get_heatmaps_gradCAM(input_model, image, class_index, layer_index):

    heatmap = []

    for idx, img in enumerate(image):

        category_index = class_index[idx]
        model_op = input_model.outputs[0]

        nb_classes = 8

        def target_layer(x): return target_category_loss(x, category_index, nb_classes)
        lyr = Lambda(target_layer,
                     output_shape=target_category_loss_output_shape)
        op = lyr([model_op])

        model = Model(inputs=input_model.input, outputs=op)
        loss = K.sum(model.layers[-1].output, -1)

        conv_output = model.layers[layer_index].output
        grads = normalize(K.gradients(loss, conv_output)[0])
        gradient_function = K.function([model.layers[0].input], [conv_output, grads])

        img = np.expand_dims(img, 0)
        output, grads_val = gradient_function([img])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        #         cam = output * grads_val

        cam = np.zeros(output.shape[0: 2], dtype=np.float64)
        weights = np.sum(grads_val, axis=(0, 1))/(grads_val.shape[0]*grads_val.shape[1])
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = np.maximum(cam, 1e-30)
        cam = cam/np.max(cam)

        cam = cv2.resize(cam, image_shape, cv2.INTER_LINEAR)
        heatmap.append(cam)

    heatmap = np.stack(heatmap, axis=0)

    return heatmap
