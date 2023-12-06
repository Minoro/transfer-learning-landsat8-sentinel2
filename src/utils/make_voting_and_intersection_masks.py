import sys
import os
import numpy as np
import rasterio
from glob import glob
from tqdm.auto import tqdm

IMAGES_STACK_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/stack_20m'
MASKS_PATH = '../../resources/sentinel/images/methods'
OUTPUT_PATH = '../../resources/sentinel/images/methods'

# Methods to be combined
# Accept: 'KatoNakamura'. 'Liu' and 'Murphy'
METHODS = [
    'KatoNakamura', 'Liu', 'Murphy'
]

NUM_VOTES = 2

images_paths = glob(os.path.join(IMAGES_STACK_PATH, '*.tif'))


for image_path in tqdm(images_paths, total=len(images_paths)):

    image_name = os.path.basename(image_path)    
    mask_name = image_name.replace('_stack.tif', '_mask.tif').replace('_20m', '')

    with rasterio.open(image_path) as src:
        meta = src.meta

    final_mask = np.zeros((meta['height'], meta['width']))
    
    meta.update(nodate=None)
    meta.update(count=1)
    meta.update(dtype=rasterio.uint8)

    for method in METHODS:
        mask_path = os.path.join(MASKS_PATH, method, mask_name)
        with rasterio.open(mask_path) as src:
            
            final_mask += (src.read(1) > 0)

    # Máscara de votação
    output_path = os.path.join(OUTPUT_PATH, 'Voting')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, mask_name)
    with rasterio.open(output_path, 'w+', **meta) as dst:
        dst.write_band(1, (final_mask >= NUM_VOTES))

    output_path = os.path.join(OUTPUT_PATH, 'Intersection')
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, mask_name)
    with rasterio.open(output_path, 'w+', **meta) as dst:
        dst.write_band(1, (final_mask == len(METHODS)))


print('Done')