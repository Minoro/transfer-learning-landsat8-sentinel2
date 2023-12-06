import os
import numpy as np
from glob import glob
import rasterio 

ANNOTAITONS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/annotations/mask1'
OUTPUT_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/annotations/mask1_array'

if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    annotations = glob(os.path.join(ANNOTAITONS_PATH, '*.tif'))

    for annotation in annotations:
        with rasterio.open(annotation) as src:
            data = src.read(1)

        annotation_name = os.path.basename(annotation)
        output_file = os.path.join(OUTPUT_PATH, annotation_name.replace('.tif', '.txt'))
        
        print(f'Annotation: {annotation_name} - Fire: {data.sum()}')
        print(data.shape)
        np.savetxt(output_file, (data).astype(int), fmt='%i')

    print('Done!')