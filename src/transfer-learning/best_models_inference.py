import sys
import os
# CUDA_DEVIDE = '-1'
CUDA_DEVIDE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVIDE)
sys.path.append('../')

import numpy as np
# import rasterio
# from rasterio.enums import Resampling
from glob import glob
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import json
from sklearn.utils import shuffle
import rasterio

from image.sentinel import make_nodata_mask
from core.data import  open_image_and_mask
from core.metrics import dice_coef

MASK_LEVEL = 'mask1'
METRIC = 'F-score'

# File with the definition of the folds
FOLDS_FILE = f'../../resources/sentinel/manual_annotation_5folds_patches_8020_{MASK_LEVEL}.csv'

# The name of the models to be generated the predictions
MODELS = ['5fold_8020_mask1_unet64f_bce_augmentation']

MODELS_PATH = '../../resources/transfer_learning/output/weights'
RESULTS_MODELS_PATH = '../../resources/transfer_learning/output/results'
OUTPUT_DIR = '../../resources/transfer_learning/output/images/predictions_best_network'

LOSS_FUNCTION='bce'
QUANTIFICATION_VALUE = 10000.0

IMAGES_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/256/imgs'
MASKS_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/256/{MASK_LEVEL}'

# Strategies that will be considered to generate the images
CONFIGS = [
    'unfreeze', 'freeze_all', 'freeze_encoder'
]


def read_results_of_pretrained_models_from_path(path):
    folds = sorted(os.listdir(path))
    print(f'Num. Folds: {len(folds)}')

    data = []
    for fold in folds:

        pretrained_models = os.listdir(os.path.join(path, fold))        
        for pretrained_model in pretrained_models:
            
            for config in CONFIGS:

                if config == 'random':
                    continue

                metrics_file = os.path.join(path, fold, pretrained_model, f'{config}_results.json')

                with open(metrics_file) as f:
                    metrics = json.load(f)

                row = {
                    'fold': fold,
                    'config': config,
                    'pretrained_model': pretrained_model,
                }

                for metric in metrics:
                    row[metric] = metrics[metric]

                data.append(row)


    df = pd.DataFrame(data)
    df['fold'] = df['fold'].apply(pd.to_numeric)
    df = df.drop(['tn',	'fp', 'fn',	'tp'], axis=1)
    df = df.fillna(0)

    return df


def predict_images_and_save_to_dir(unet, images_path, masks_path, output_dir):
    images_path = sorted(images_path)
    masks_path = sorted(masks_path)

    for image_path, mask_path in tqdm(zip(images_path, masks_path), total=len(images_path)):
        image_name = os.path.basename(image_path)
        output_mask = os.path.join(output_dir, image_name)

        if os.path.exists(output_mask):
            continue
            
        with rasterio.open(image_path) as src:
            meta = src.meta

        image, mask = open_image_and_mask(tf.convert_to_tensor((image_path, mask_path)))
        image = image / QUANTIFICATION_VALUE
        prediction = unet(np.asarray([image]))[0]
        prediction = np.array(prediction > 0.5, dtype=np.uint8)
        valid_mask = ~make_nodata_mask(image)
        prediction = prediction & valid_mask
        # prediction = prediction * 255

        meta.update(count=1)
        meta.update(nodata=None)
        meta.update(dtype=rasterio.uint8)
        with rasterio.open(output_mask, 'w', **meta) as dst:
            dst.write_band(1, (prediction[:,:,0]).astype(rasterio.uint8))   


if __name__ == '__main__':
    df = pd.read_csv(FOLDS_FILE)
    df['images_paths'] = df['image'].apply(lambda x : os.path.join(IMAGES_DIR, x))
    df['masks_paths'] = df[f'{MASK_LEVEL}'].apply(lambda x : os.path.join(MASKS_DIR, x))

    
    for model in tqdm(MODELS, total=len(MODELS)):

        results_path = os.path.join(RESULTS_MODELS_PATH, model)
        df_results = read_results_of_pretrained_models_from_path(results_path)
        
        pretrained_models = df_results.pretrained_model.unique()
        
        for config in CONFIGS:
            filter_config = (df_results['config'] == config)
            best_score = df_results[ filter_config ][METRIC].max()

            filter_metric = (df_results[METRIC] == best_score)
            
            best_model = df_results[ (filter_config & filter_metric) ]
            if len(best_model) == 0:
                continue
            
            pretrained_model = best_model['pretrained_model'].values[0]
            fold = best_model['fold'].values[0]
            df_train = df[ (df['fold'] == fold) & (df['set'] == 'train')]
            df_test = df[ (df['fold'] == fold) & (df['set'] == 'test')]

            print(f'Prediction with model: {model} - config: {config} - Fold: {fold} - Score: {best_score} - Pretrained: {pretrained_model}')
            
            # Load the train patches (validation included)
            images = df_train.sentinel_image.unique()
            train_images_paths = []
            train_masks_paths = []
            for image in images:
                train_images_paths += glob(os.path.join(IMAGES_DIR, f'{image}*.tif'))
                train_masks_paths += glob(os.path.join(MASKS_DIR, f'{image}*.tif'))

            train_images_paths = sorted(train_images_paths)
            train_masks_paths = sorted(train_masks_paths)


            test_images_paths = []
            test_masks_paths = []
            images = df_test.sentinel_image.unique()
            for image in images:
                test_images_paths += glob(os.path.join(IMAGES_DIR, f'{image}*.tif'))
                test_masks_paths += glob(os.path.join(MASKS_DIR, f'{image}*.tif'))

            test_images_paths = sorted(test_images_paths)
            test_masks_paths = sorted(test_masks_paths)

            # clear the memory
            tf.keras.backend.clear_session()

            unet = tf.keras.models.load_model(
                os.path.join(MODELS_PATH, model, f'{pretrained_model}_fold_{fold}_nsamples_{config}_{LOSS_FUNCTION}'),
                custom_objects={'dice_coef': dice_coef}
            )
            unet.summary()

            output_dir = os.path.join(OUTPUT_DIR, model, pretrained_model, f'fold_{fold}', 'train', config)
            os.makedirs(output_dir, exist_ok=True)
            print('Predicting traning samples')
            predict_images_and_save_to_dir(unet, train_images_paths, train_masks_paths, output_dir)


            output_dir = os.path.join(OUTPUT_DIR, model, pretrained_model, f'fold_{fold}', 'test', config)
            os.makedirs(output_dir, exist_ok=True)
            print('Predicting test samples')
            predict_images_and_save_to_dir(unet, test_images_paths, test_masks_paths, output_dir)

            print(f'Prediction done for model {model} - Config: {config} - Fold: {fold} - Score: {best_score} - Pretrained: {pretrained_model}')
