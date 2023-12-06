import sys
sys.path.append('../')

from image.sentinel import make_nodata_mask

# from image.converter import get_gml_geometry
from active_fire.general import ActiveFireIndex
from utils.metadata import get_image_metadata

from tqdm import tqdm
import rasterio
import os
import numpy as np
from glob import glob

# import cv2

IMAGES_STACK_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/stack_20m'
METADATA_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/metadata'

OUTPUT_DIR = '../../resources/sentinel/images/output_txt'


# If True save the output as txt files, if False save as TIF images
SAVE_AS_TXT = False


# Methods to generate the active fire masks, the methods will be instantiate by name (provided by key 'method')
# The masks will be generated using the 'transform' method, the values in the 'args' key will be passed to the method  
# The Murphy's method need the TOA corrected by the solar zenith, therefore the argument 'apply_solar_zenith_correction'
# is informed to the method in order to perform the correction
# The methods can be "Liu", "KatoNakamura" or "Murphy"
ALGORITHMS = [
    {'method': 'Liu', 'args': {}},
    {'method': 'KatoNakamura', 'args': {}},
    {'method': 'Murphy', 'args': {'apply_solar_zenith_correction': True}},
]


# Name of the stacks that will be used, leave empty to process all files in the IMAGES_STACK_DIR
STACKS = [
    #Example:
    # 'T12TUQ_20200813T181931',
    # 'T16SCD_20200815T162839',
]


def get_stack_names():    
    """Get the name of the stacks that will be processed.
    Returns:
        list: Name of the stacks to process
    """
    stacks = STACKS
    if len(STACKS) == 0:
        stacks = ['_'.join(stack.split('_')[:2]) for stack in os.listdir(IMAGES_STACK_DIR)]
    return stacks

def read_metadata_file(stack_name):
    """Read the metadata for a image stack 

    Args:
        stack_name (str): Base name of the image

    Returns:
        dict: metadata in a key-value format
    """
    metadata_dir = os.path.join(METADATA_DIR, stack_name)

    metadata_mtl = os.path.join(metadata_dir, 'MTD_TL.xml')
    metadata_msil = os.path.join(metadata_dir, 'MTD_MSIL1C.xml')
    metadata = get_image_metadata(metadata_mtl, metadata_msil)
    return metadata

def get_algorithms():
    """Get the algorithms that will be used to generate the active fire mask
    To generated the mask the transform method can be used

    Returns:
        list: List with the altorithms
    """
    algorithms = []
    
    for algorithm in ALGORITHMS:
        alg = algorithm.copy()
        alg['afi'] = ActiveFireIndex(alg['method'])

        algorithms.append(alg)

    return algorithms

if __name__ == '__main__':
    
    if SAVE_AS_TXT:
        print(f'[INFO] Output saved as TXT files at: {OUTPUT_DIR}')
    else:
        print(f'[INFO] Output saved as GeoTIF files at: {OUTPUT_DIR}')

    stack_names = get_stack_names()
    algorithms = get_algorithms()

    for stack_name in tqdm(stack_names):

        metadata = read_metadata_file(stack_name)
        
        # Read the NIR, SWIR1, SWIR2 TOA reflectance for the image
        stack_path = os.path.join(IMAGES_STACK_DIR, f'{stack_name}_20m_stack.tif')
        with rasterio.open(stack_path) as src:
            img = src.read((4,5,6)).transpose((1,2,0)) / 10000.0
            meta = src.meta
        

        for algorithm in algorithms:
            afi = algorithm['afi']

            # Generate the masks to the algorithm
            args = algorithm.get('args', {})
            mask = afi.transform(img, metadata=metadata, **args)
            valid_mask = ~make_nodata_mask(img)
            mask = mask & valid_mask
            # output_dir = os.path.join(OUTPUT_DIR, method, stack_name.replace('.tif', ''))
            output_folder = algorithm.get('save_as', algorithm['method'])
            output_dir = os.path.join(OUTPUT_DIR, output_folder)    
            os.makedirs(output_dir, exist_ok=True)

            if SAVE_AS_TXT:
            
                file_path = os.path.join(output_dir, 'det_{}.txt'.format(stack_name.replace('.tif', '')))

                # Save as TXT
                np.savetxt(file_path, (mask).astype(int), fmt='%i')
                # print(np.loadtxt(file_path).shape)

            else:
                # Save as png
                # mask = mask * 255
                # cv2.imwrite(os.path.join(output_dir, '{}_mask.png'.format(stack_name)), mask)

                # Save as TIF
                meta.update(count=1)
                meta.update(nodata=None)
                output_mask = os.path.join(output_dir, 'det_{}_mask.tif'.format(stack_name))
                with rasterio.open(output_mask, 'w', **meta) as dst:
                    dst.write_band(1, (mask).astype(rasterio.uint16))   





