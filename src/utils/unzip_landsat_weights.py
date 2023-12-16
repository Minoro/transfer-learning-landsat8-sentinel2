import sys
import os
import zipfile


LANDSAT_ZIPED_WEIGHTS = '../../resources/landsat_weights.zip'
OUTPUT_PATH = '../../resources/'


if __name__ == '__main__':
    
    print('Unziping Landsat-8 Weights')

    with zipfile.ZipFile(LANDSAT_ZIPED_WEIGHTS) as zip:
        zip.extractall(OUTPUT_PATH)

    print('Done!')
