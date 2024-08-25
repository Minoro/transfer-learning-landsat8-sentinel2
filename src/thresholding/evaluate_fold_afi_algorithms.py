import sys
sys.path.append('../')
import os
import numpy as np
import glob
import pandas as pd
import rasterio
import json
from tqdm.auto import tqdm

from core.metrics import evaluate


RANDOM_STATE = 42

# Extension of the annotations and the predictions
PREDICTION_FORMAT = 'tif'

ANNOTATIONS_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/annotations/mask1'
PREDICTIONS_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/methods'

ANNOTATION_NAME_IDENTIFICATION = '_20m_stack_maskf'
PREDICTION_NAME_IDENTIFICATION = '_mask'

DATAFRAME_FOLDS_PATH = f'../../resources/sentinel/manual_annotation_5folds_patches_8020_mask1.csv'

RESULTS_OUTPUT_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/evaluate_algorithms_results/5folds_mask1'


if __name__ == '__main__':

    algorithms = os.listdir(PREDICTIONS_DIR)

    if len(algorithms) == 0:
        print('[ERROR] Masks of thresholding algorithms not found.')
        sys.exit()

    df = pd.read_csv(DATAFRAME_FOLDS_PATH)
    folds = df.fold.unique()

    # By default the test set is equals for all folds. 
    # Therefore the result will be the same, since the thresholding methods has no "randomness".
    for fold in folds:
        print('Evaluating K:', fold)
                
        result_output_dir = os.path.join(RESULTS_OUTPUT_DIR, str(fold))
        os.makedirs(result_output_dir, exist_ok=True)
        

        # The "folds" are made with the whole image
        test_images = df[ (df['set'] == 'test') & (df['fold'] == fold)]['sentinel_image'].unique()

        annotations_paths = []
        for test_image in test_images:
            annotations_paths += glob.glob(os.path.join(ANNOTATIONS_DIR, f'{test_image}*.{PREDICTION_FORMAT}')) 

        annotations_paths = sorted(annotations_paths) 


        # Load all annotations in memory
        print('Loading manual anotations...')
        annotations = []
        for annotation_path in annotations_paths:
            with rasterio.open(annotation_path) as src:
                annotation = (src.read(1) != 0)

            annotations.append(annotation)
        annotations = np.array(annotations)
        print('Manual anotations loaded!')


        for algorithm in algorithms:
            print(f'Evaluating {algorithm} - Fold: {fold}')
            predictions_paths = []
            for test_image in tqdm(test_images, total=len(test_images)):
                predictions_paths += glob.glob(os.path.join(PREDICTIONS_DIR, algorithm, f'{test_image}*.{PREDICTION_FORMAT}'))

            predictions_paths = sorted(predictions_paths)
            predictions = []
            for annotation_path, prediction_path in zip(annotations_paths, predictions_paths):

                annotation_name = os.path.basename(annotation_path)
                pred_name = os.path.basename(prediction_path)

                # Make sure that the annotation and the prediction are from the same image
                if annotation_name.replace(ANNOTATION_NAME_IDENTIFICATION, PREDICTION_NAME_IDENTIFICATION) != pred_name:
                    print(f'[ERROR] The annotation and the prediciton dont match - Annotation: {annotation_name} - Prediction: {pred_name} - Method: {algorithm}')
                    sys.exit()

                with rasterio.open(prediction_path) as src:
                    pred = (src.read(1) != 0)

                predictions.append(pred)


            print('Checking: {}'.format(algorithm))
            print('# Masks: {}'.format(len(annotations)))
            print('# Pred.: {}'.format(len(predictions)))
            
            # annotations = np.array(annotations)
            predictions = np.array(predictions)

            # Evaluate the performance
            result = evaluate(annotations, predictions)
            print(result)

            with open(os.path.join(RESULTS_OUTPUT_DIR, str(fold), algorithm + '.json'), 'w') as outfile:
                json.dump(result, outfile)
