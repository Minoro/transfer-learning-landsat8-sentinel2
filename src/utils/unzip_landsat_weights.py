import sys
import os
import zipfile
import shutil

LANDSAT_ZIPED_WEIGHTS = '../../resources/landsat/weights/unet/landsat_weights.zip'
OUTPUT_PATH = '../../resources/landsat/weights/unet'

WEIGHTS_NAME_PATTERNS = 'model_unet_{}_765_final_weights.h5'

MASKS_NAMES = ['Kumar-Roy', 'Schroeder', 'Murphy', 'Intersection', 'Voting']

if __name__ == '__main__':
    
    print('Unziping Landsat-8 Weights...')

    with zipfile.ZipFile(LANDSAT_ZIPED_WEIGHTS) as zip:
        zip.extractall(OUTPUT_PATH)

    print('Weights unziped!')
    print('Moving weights...')
    for mask_name in MASKS_NAMES:
        
        weights_name = WEIGHTS_NAME_PATTERNS.format(mask_name)
        weights_path = os.path.join(OUTPUT_PATH, weights_name)
        if not os.path.exists(weights_path):
            continue

        output_path = os.path.join(OUTPUT_PATH, mask_name, 'B7B6B5')
        os.makedirs(output_path, exist_ok=True)
        
        shutil.move(weights_path, output_path)

    print('Weights moved!')

    print('Done!')
