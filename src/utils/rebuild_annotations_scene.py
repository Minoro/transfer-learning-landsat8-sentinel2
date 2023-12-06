import os
import sys
import rasterio
from glob import glob

STACKS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/stack'
CROPED_ANNOTATIONS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/256/mask1'
OUTPUT_SCENE_ANNOATIONS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/annotations/mask1'
ANNOTATION_SUFIX = 'maskf'

if __name__ == '__main__':
    """ Transform the croped 256x256 manual annotations back to original size in one single file
    """
    os.makedirs(OUTPUT_SCENE_ANNOATIONS_PATH, exist_ok=True)

    stacks = glob(os.path.join(STACKS_PATH, '*_20m_stack.tif'))

    for stack in stacks:

        print('Stack: ', stack)
        stack_name = '_'.join(os.path.basename(stack).split('_')[:2])
        masks = glob(os.path.join(CROPED_ANNOTATIONS_PATH, f'{stack_name}*.tif'))

        with rasterio.open(stack) as src:
            meta = src.meta

        meta.update(nodata=None)
        meta.update(count=1)
        output_mask = f'{stack_name}_{ANNOTATION_SUFIX}.tif'
        with rasterio.open(os.path.join(OUTPUT_SCENE_ANNOATIONS_PATH, output_mask), 'w+', **meta) as dst:

            for mask_path in masks:
                with rasterio.open(mask_path) as src:
                    # Compute the position on the original scene using the geolocation
                    bounds = rasterio.transform.array_bounds(src.meta['height'], src.meta['width'], src.transform) 
                    window = rasterio.windows.from_bounds(*bounds, transform=meta['transform'])   

                    # write the final mask 
                    dst.write_band(1, src.read(1), window=window)


    print('Done!')