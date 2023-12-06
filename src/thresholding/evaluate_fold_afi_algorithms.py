import sys
sys.path.append('../')
import os
import numpy as np
import glob
import pandas as pd
import rasterio
import json

from core.metrics import evaluate


RANDOM_STATE = 42

# Extension of the annotations and the predictions
# It can be txt or tif
PREDICTION_FORMAT = 'txt'

PREDICTIONS_DIR = '../../resources/sentinel/images/output_txt'
ANNOTATIONS_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/scenes/annotations/mask1_array'

DATAFRAME_FOLDS_PATH = f'../../resources/sentinel/manual_annotation_5folds_patches_8020_mask1.csv'


RESULTS_OUTPUT_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/evaluate_algorithms_results/5folds_mask1_array'


if __name__ == '__main__':

    algorithms = os.listdir(PREDICTIONS_DIR)

    df = pd.read_csv(DATAFRAME_FOLDS_PATH)
    folds = df.fold.unique()
    
    test_tiles = df[ (df['set'] == 'test') ]['sentinel_image'].unique()
    
    for fold in folds:
        print('Evaluating K:', fold)
                
        result_output_dir = os.path.join(RESULTS_OUTPUT_DIR, str(fold))
        os.makedirs(result_output_dir, exist_ok=True)
        
        test_tiles = df[ (df['set'] == 'test') & (df['fold'] == fold)]['sentinel_image'].unique()
        
        # Find the manual annotated files
        txts_mask_path = []
        for test_tile in test_tiles:
            txts_mask_path += glob.glob(os.path.join(ANNOTATIONS_DIR, f'{test_tile}*.{PREDICTION_FORMAT}')) 
        

        txts_mask_path = sorted(txts_mask_path)
        for algorithm in algorithms:
            # Load the predictions of the algorithms
            txts_pred_path = []
            for test_tile in test_tiles:
                txts_pred_path += glob.glob(os.path.join(PREDICTIONS_DIR, algorithm, f'det_{test_tile}*.{PREDICTION_FORMAT}')) 
            
            txts_pred_path = sorted(txts_pred_path)

            # Load all images and annotations to one "big" array to be compared
            masks = []
            predictions = []
            for txt_mask_path, txt_pred_path in zip(txts_mask_path, txts_pred_path):

                mask_name = os.path.basename(txt_mask_path).replace(f'.{PREDICTIONS_DIR}', '')
                pred_name = os.path.basename(txt_pred_path).replace(f'.{PREDICTIONS_DIR}', '')

                mask_name = '_'.join(mask_name.split('_')[:2])
                pred_name = '_'.join(pred_name.split('_')[1:3])
                
                if mask_name != pred_name:
                    print('[ERROR] A máscara e a predição não coincidem')
                    sys.exit()
                
                if PREDICTION_FORMAT == 'txt':
                    mask = np.loadtxt(txt_mask_path)
                    pred = np.loadtxt(txt_pred_path)

                elif PREDICTION_FORMAT == 'tif':
                    with rasterio.open(txt_mask_path) as src:
                        mask = (src.read(1) != 0)

                    with rasterio.open(txt_pred_path) as src:
                        pred = (src.read(1) != 0)

                masks.append(mask)
                predictions.append(pred)

            print('Checking: {}'.format(algorithm))
            print('# Masks: {}'.format(len(masks)))
            print('# Pred.: {}'.format(len(predictions)))
            

            masks = np.array(masks)
            predictions = np.array(predictions)

            # Evaluate the performance
            result = evaluate(masks, predictions)
            print(result)

            with open(os.path.join(RESULTS_OUTPUT_DIR, str(fold), algorithm + '.json'), 'w') as outfile:
                json.dump(result, outfile)