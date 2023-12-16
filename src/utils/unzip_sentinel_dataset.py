import sys
import os
import zipfile

UNZIPING_SCENE = True

SENTINEL_ZIPED_PATCHES = '../../resources/dataset_patches.zip'
SENTINEL_ZIPED_SCENES = '../../resources/dataset_scenes.zip'

OUTPUT_PATH = '../../resources/sentinel/Sentinel2/manual_annotated'


if __name__ == '__main__':

    zip_path = SENTINEL_ZIPED_PATCHES
    if UNZIPING_SCENE:
        zip_path = SENTINEL_ZIPED_SCENES

    print('Unzip Sentinel-2 dataset...')

    with zipfile.ZipFile(zip_path) as zip:
        zip.extractall(OUTPUT_PATH)

    print('Done!')

