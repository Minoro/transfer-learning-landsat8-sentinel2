import sys
import os
import json
import numpy as np

import tensorflow as tf
import argparse
from datetime import datetime

sys.path.append('../')
from core.models import get_model, get_models_available
import landsat.landsat_config as config




import os
import warnings
warnings.filterwarnings("ignore")

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
# from tensorflow.keras.backend.tensorflow_backend import set_session
import tensorflow.keras as keras
# from keras import optimizers
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers

from metrics import *

import sys
import pandas as pd
import json

from tqdm import tqdm

import rasterio

MANUAL_ANNOTATION_PATH = '../../../../../dataset/manual_annotations/patches/manual_annotations_patches'
IMAGES_PATH = '../../../../../dataset/manual_annotations/patches/landsat_patches'

CUDA_DEVICE = 0

# 10 or 3
N_CHANNELS = 3

MASK_ALGORITHM = 'Intersection'
MODEL_NAME = 'deeplabv3+'
WEIGHTS_FILE = './weights/model_{}_{}_765_final_weights.h5'.format(MODEL_NAME, MASK_ALGORITHM)


MASKS_ALGORITHMS = ['Schroeder', 'Murphy', 'Kumar-Roy', 'Intersection', 'Voting']

# Desconsideras as mascaras que contem os seguintes itens no nome
IGNORE_MASKS_WITH_STR = ['v2']

# Remove as strings definidas do nome das mascas para realizar um match entre mascara e imagem analisada
REMOVE_STR_FROM_MASK_NAME = MASKS_ALGORITHMS + ['v1']


TH_FIRE = 0.5
IMAGE_SIZE = (256, 256)
MAX_PIXEL_VALUE = 65535 # Max. pixel value, used to normalize the image

OUTPUT_DIR = './log'

def load_path_as_dataframe(mask_path):
    masks = glob(os.path.join(mask_path, '*.tif'))

    print('Carregando diretório: {}'.format(mask_path))
    print('Total de máscaras no diretórios: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['masks_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.masks_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MASKS_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df

def remove_algorithms_name(mask_name):
    """Remove o nome dos algoritmos do nome da máscara"""

    #algorithms_name = MASKS_ALGORITHMS + ['mask' , 'noFire']
    # algorithms_name = MASKS_ALGORITHMS + ['v2' , 'noFire']

    for algorithm in REMOVE_STR_FROM_MASK_NAME:
        mask_name = mask_name.replace('_{}'.format(algorithm), '')

    return mask_name

def merge_dataframes(df_manual, df_images):
    return pd.merge(df_manual, df_images, on = 'image_name', how='outer')



def open_manual_annotation(path):
    if type(path) != str:
        mask = np.zeros(IMAGE_SIZE)
    else:
        mask = get_mask_arr(path)

    return np.array(mask, dtype=np.uint8)

def get_mask_arr(path):
    """ Abre a mascara como array"""
    with rasterio.open(path) as src:
        img = src.read().transpose((1, 2, 0))
        seg = np.array(img, dtype=int)

        return seg[:, :, 0]


def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img


def get_img_765bands(path):
    img = rasterio.open(path).read((7,6,5)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img




# if __name__ == '__main__':
    
#     if not os.path.exists(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays')):
#         os.makedirs(os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays'))

#     os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)

#     # Define a função de abertura de imagens
#     open_image = get_img_arr
#     if N_CHANNELS == 3:
#         open_image = get_img_765bands


#     model = DeeplabV3Plus(image_size=IMAGE_SIZE[0], num_classes=1)

#     model.summary()
#     print('Loading weghts...')
#     model.load_weights(WEIGHTS_FILE)
#     print('Weights Loaded')


#     df_manual = load_path_as_dataframe(MANUAL_ANNOTATION_PATH)
#     df_images = load_path_as_dataframe(IMAGES_PATH)

#     df = merge_dataframes(df_manual, df_images)
#     print('Total de Imagens: {}'.format(len(df.index)))


#     for index, row in tqdm(df.iterrows(), total=len(df)):
        
#         y_true = open_manual_annotation(row['masks_path_x'])
#         img = open_image(row['masks_path_y'])


#         y_pred = model.predict(np.array( [img] ), batch_size=1)
#         y_pred = y_pred[0, :, :, 0] > TH_FIRE
#         y_pred = np.array(y_pred, dtype=np.uint8)
        
#         image_path = row['image_name']

#         mask_name = os.path.splitext(os.path.basename(image_path))[0]
#         image_name = os.path.splitext(os.path.basename(image_path))[0]

#         txt_mask_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'grd_' + mask_name + '.txt') 
#         txt_pred_path = os.path.join(OUTPUT_DIR, MASK_ALGORITHM, 'arrays', 'det_' + image_name + '.txt') 

#         np.savetxt(txt_mask_path, y_true.astype(int), fmt='%i')
#         np.savetxt(txt_pred_path, y_pred.astype(int), fmt='%i')


def make_predictions_using_model(model):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use a trained model to predict active fire in Landsat-8 images.')
    parser.add_argument('--model', action="store", choices=get_models_available(), required=True, help="Trained model to be used")
    parser.add_argument('--mask', action="store", choices=os.listdir(config.IMAGES_DATAFRAMES_PATH), required=True, help="Mask used to train the base model")
    parser.add_argument('--txt', action="store_true", help="Save the predictions as TXT with 0 and 1, otherwise save the predictions as npz files")
    
    args = parser.parse_args()
    weights_dir = os.path.join(config.LANDSAT_OUTPUT_DIR, args.model, args.mask)
    model_weights_path = os.path.join(weights_dir, config.FINAL_WEIGHTS_OUTPUT.format(args.model, args.mask))

    # print('Loading weghts...')
    # model = get_model(args.model, input_shape=config.IMAGE_SHAPE, num_classes=1)
    # model.load_weights(model_weights_path)
    # print('Weights Loaded')
    # print(args)
    # model.summary()

    masks = glob(os.path.join(, '*.tif'))

    print('Carregando diretório: {}'.format(mask_path))
    print('Total de máscaras no diretórios: {}'.format(len(masks)))

    df = pd.DataFrame(masks, columns=['masks_path'])
    # recupera o nome da máscara pelo caminho dela
    df['original_name'] = df.masks_path.apply(os.path.basename)
    # remove o algoritmo gerador da mascara do nome dela
    df['image_name'] = df.original_name.apply(remove_algorithms_name)

    # remove mascaras com as strings definidas
    for ignore_mask_with_str in IGNORE_MASKS_WITH_STR:
        df = df[~df.original_name.str.contains(ignore_mask_with_str)]

    return df