import sys
import os
import zipfile

SENTINEL_ZIPED_PATCHES = '/hd1/andre/JSTARS/transfer-learning-landsat8-sentinel2/manual_annotated_croped_and_scene.zip'
# SENTINEL_ZIPED_PATCHES = '../../resources/sentinel/sentinel2_manual_annotated_croped_and_scene.zip'
OUTPUT_PATH = '../../resources/Sentinel2'


if __name__ == '__main__':

    zip_path = SENTINEL_ZIPED_PATCHES

    print('Unzip Sentinel-2 dataset...')

    with zipfile.ZipFile(zip_path) as zip:
        zip.extractall(OUTPUT_PATH)

    print('Done!')

