import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from tqdm import tqdm
from image.sentinel import make_nodata_mask

def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.clip(y_true, -1e12, 1e12)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = K.clip(y_pred, -1e12, 1e12)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    
    intersection = K.sum(y_true_f * y_pred_f)    
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def jaccard(y_true, y_pred, smooth=100):
    y_true = K.clip(y_true, -1e12, 1e12)
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = K.clip(y_pred, -1e12, 1e12)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (jac) * smooth / 100.0



def statistics3 (y_true, y_pred):
    y_pred_neg = 1 - y_pred
    y_expected_neg = 1 - y_true

    tp = np.sum(y_pred * y_true)
    tn = np.sum(y_pred_neg * y_expected_neg)
    fp = np.sum(y_pred * y_expected_neg)
    fn = np.sum(y_pred_neg * y_true)
    return tn, fp, fn, tp


def pixel_accuracy (y_true, y_pred):
    sum_n = np.sum(np.logical_and(y_pred, y_true))
    sum_t = np.sum(y_true)
 
    if (sum_t == 0):
        pixel_accuracy = 0
    else:
        pixel_accuracy = sum_n / sum_t
    return pixel_accuracy    


def loss_function_factory(loss_fn_name):
    loss_fn_name = loss_fn_name.lower()
    if loss_fn_name == 'bce':
        def loss_function(y_true, y_pred):
            y_pred = K.clip(y_pred, -1e12, 1e12)
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)

            loss_fn = tf.keras.losses.BinaryCrossentropy()
            loss = loss_fn(y_true, y_pred)
            return loss
    elif loss_fn_name == 'jaccard':
        def loss_function(y_true, y_pred, smooth=100):
            y_pred = K.clip(y_pred, -1e12, 1e12)
            y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)

            intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
            sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
            jac = (intersection + smooth) / (sum_ - intersection + smooth)
            
            return (1 - jac) * smooth
    else:
        raise ValueError('Loss function unknown. Use "bce" or "jarccard".')

    return loss_function

def evaluate(annotations, predictions):

    if len(annotations) != len(predictions):
        raise 'Número de predições e anotações diferente'

    y_pred_all_v1 = []
    y_true_all_v1 = []

    # jaccard_score_sum_v1 = 0
    # f1_score_sum_v1 = 0
    # pixel_accuracy_sum_v1 = 0

    count_fire_pixel_mask = 0
    count_fire_pixel_pred = 0
    nsum_v1 = 0

    i = 0
    for y_true, y_pred in tqdm(zip(annotations, predictions), total=len(predictions)):
        i += 1

        y_true = np.array(y_true != 0, dtype=np.uint8)
        y_pred = np.array(y_pred > 0.5, dtype=np.uint8)

        y_pred = y_pred.flatten() 
        y_true = y_true.flatten()

        y_pred_all_v1.append(y_pred)
        y_true_all_v1.append(y_true)

        nsum_v1 = nsum_v1 + 1

        count_fire_pixel_mask += np.sum(y_true)
        count_fire_pixel_pred += np.sum(y_pred)

    print('True Fire:', count_fire_pixel_mask)
    print('Fire Pred:', count_fire_pixel_pred)
    # print('Done!')


    y_pred_all_v1 = np.array(y_pred_all_v1, dtype=np.uint8)
    y_pred_all_v1 = y_pred_all_v1.flatten()

    y_true_all_v1 = np.array(y_true_all_v1, dtype=np.uint8)
    y_true_all_v1 = y_true_all_v1.flatten()

    print('y_true_all_v1 shape: ', y_true_all_v1.shape)
    print('y_pred_all_v1 shape: ', y_pred_all_v1.shape)

 
    tn, fp, fn, tp = statistics3(y_true_all_v1, y_pred_all_v1)
    print ('Statistics3 (tn, fp, fn, tp):', tn, fp, fn, tp)

    P = float(tp)/(tp + fp)
    R = float(tp)/(tp + fn)
    IoU = float(tp)/(tp+fp+fn)
    Acc = float((tp+tn))/(tp+tn+fp+fn)
    if (P + R) == 0:
        F = 0.0
    else:
        F = (2 * P * R)/(P + R)
    
    print('P: :', P, ' R: ', R, ' IoU: ', IoU, ' Acc: ', Acc, ' F-score: ', F, 'TP: ', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)

    results = {
        'P' : P,
        'R' : R,
        'IoU': IoU,
        'F-score': F,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }

    return results


def evaluate_dataset(model, dataset):

    predictions = model.predict(dataset)
    print('Número de predições: ', len(predictions))

    annotations = []
    for i, (img, annotation) in enumerate(dataset):
        for mask in annotation:
            annotations.append(mask)
        
    annotations = np.array(annotations)
    print('Número de anotações: ', len(annotations))
    print('Número de predições', len(predictions))

    return evaluate(annotations, predictions)

