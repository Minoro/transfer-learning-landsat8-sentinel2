import os
import sys
import rasterio
from glob import glob
from tqdm import tqdm

CROPED_IMAGES_PATH = '../../resources/transfer_learning/output/predictions'
STACKS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/scenes/stack'
OUTPUT_SCENE_PATH = '../../resources/transfer_learning/output/images/scene/predictions'


if __name__ == '__main__':
    """ Transform the croped 256x256 back to original size in one single file
    """
    # os.makedirs(OUTPUT_SCENE_PATH, exist_ok=True)

    patches = glob(os.path.join(CROPED_IMAGES_PATH, '**', '*.tif'), recursive=True)

    stacks_paths = []
    for patch in patches:
        
        patch_name = os.path.basename(patch)
        patch_name = patch_name.split('_')
        stack_name = '_'.join(patch_name[:2])
        stacks_paths.append(os.path.join(STACKS_PATH, f'{stack_name}_20m_stack.tif'))

    stacks = list(set(stacks_paths))

    print(len(stacks))



    for stack in stacks:

        print('Stack: ', stack)
        stack_name = '_'.join(os.path.basename(stack).split('_')[:2])


        with rasterio.open(stack) as src:
            meta = src.meta
        meta.update(nodata=None)
        meta.update(count=1)


        models = os.listdir(CROPED_IMAGES_PATH)
        # print(models)
        for model in models:
            pretrained_models = os.listdir(os.path.join(CROPED_IMAGES_PATH, model))
            
            for pretrained_model in pretrained_models:
                # folds = os.listdir(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model))
                folds = ['1']
                for fold in folds:
                    # datasets = os.listdir(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model, fold))
                    datasets = os.listdir(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model))
                    
                    for dataset in datasets:
                        
                        # configs = os.listdir(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model, fold, dataset))
                        configs = os.listdir(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model, dataset))

                        for config in configs:
                            # predictions_path = os.path.join(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model, fold, dataset, config))
                            predictions_path = os.path.join(os.path.join(CROPED_IMAGES_PATH, model, pretrained_model, dataset, config))

                            masks = glob(os.path.join(predictions_path,f'{stack_name}*.tif'))
                            print(predictions_path)
                            if len(masks) == 0:
                                break

                            
                            output_dir = os.path.join(OUTPUT_SCENE_PATH, model, config, pretrained_model, fold, dataset)
                            output_mask = os.path.join(output_dir, f'{stack_name}.tif')

                            # Ignora imagem já processada
                            if os.path.exists(output_mask):
                                continue
                            
                            os.makedirs(output_dir, exist_ok=True)

                            print(f'Processando: {output_mask}')
                            with rasterio.open(output_mask, 'w+', **meta) as dst:
                                for mask_path in tqdm(masks):
                                    with rasterio.open(mask_path) as src:
                                        # Compute the position on the original scene using the geolocation
                                        bounds = rasterio.transform.array_bounds(src.meta['height'], src.meta['width'], src.transform) 
                                        window = rasterio.windows.from_bounds(*bounds, transform=meta['transform'])   

                                        # write the final mask 
                                        dst.write_band(1, src.read(1), window=window)



    print('Done!')