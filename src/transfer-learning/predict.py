import os
import pandas as pd
from glob import glob
import tensorflow as tf
from tqdm.auto import tqdm
import argparse
import rasterio
import numpy as np

from PIL import Image

from core.models import get_normalization_layer
from core.data import get_dataset_from_paths
import config


def read_fold_file(fold):    
    return pd.read_csv(os.path.join(config.CSV_FOLDS_BASE_PATH, f'fold_{fold}.csv'))


def get_train_validation_test_dataset_using_normalization_layer(df_fold, normalization_layer):
    df_train, df_validation, df_test = filter_train_validation_test_dataframe(df_fold)

    train_dataset = get_dataset_from_paths(df_train['images_paths'], df_train['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=config.USE_DATA_AUGMENTATION, shuffle=True)
    val_dataset = get_dataset_from_paths(df_validation['images_paths'], df_validation['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False)
    test_dataset = get_dataset_from_paths(df_test['images_paths'], df_test['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False, repeat=False)

    return train_dataset, val_dataset, test_dataset


def filter_train_validation_test_dataframe(df_fold):    
    df_fold['images_paths'] = df_fold.apply(build_image_path, axis=1)
    df_fold['masks_paths'] = df_fold.apply(build_mask_path, axis=1)

    df_train = df_fold[(df_fold['set'] == 'train')]
    df_validation = df_fold[(df_fold['set'] == 'validation')]
    df_test = df_fold[(df_fold['set'] == 'test')]

    return df_train, df_validation, df_test

def build_image_path(row):
    return os.path.join(config.SENTINEL_MANUAL_ANNOTATIONS_IMAGES_PATH, row['patch'])
    
def build_mask_path(row):
    return os.path.join(config.SENTINEL_MANUAL_ANNOTATIONS_MASK_PATH, row['patch'].replace(config.ANNOTATION_IMAGE_MARKER,  config.ANNOTATION_MASK_MARKER))
    
def decode_png_patch_name(model_name, identification, patch_name):
    
    if 'unet' == model_name:
        model_name = 'U-net'

    tl_strategy = 'Basic_TL-NoFT_NoBN'
    if 'TL_with_FT' in identification:
        tl_strategy = 'TL_with_FT_NoBN'
    
    
    png_name = patch_name.replace('_20m_stack_p', '_20m_stack_s256_p')
    png_name = png_name.replace('.tif', f'_{model_name}_{tl_strategy}.png')

    return png_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gera as predições para o conjunto de teste dos modelos ajustados para o Sentinel-2.')
    
    parser.add_argument('--model', action="store", choices=config.PRETRAINED_MODELS, default=[config.MODEL], nargs='*', help="Modelo(s) para passar(em) por Transfer Learning" )  
    parser.add_argument('--fold', action="store", choices=list(range(1, config.NUM_FOLDS+1)), default=list(range(1, config.NUM_FOLDS+1)), type=int, nargs='*', help="Número(s) do(s) fold(s) para ser(em) avaliado(s)")
    parser.add_argument('--normalization', action="store", choices=['bn', 'no-bn', None], default=config.NORMALIZATION_MODE, help="Modo de normalização da imagem de entrada da rede")
    parser.add_argument('--quantification', action="store", type=float, default=config.SENTINEL_QUANTIFICATION_VALUE, help='Valor de quantificação das imagens do Sentinel-2. Se houver normalização, a imagem será dividida por esse valor')
    
    parser.add_argument('--png', action="store_true", help="Salva as predições em PNG")
    parser.add_argument('--gpu', action="store", type=str, default=str(config.CUDA_DEVICE), help="Dispositivo GPU usado")
    parser.add_argument('--identification', action="store",  type=str, nargs='*', help="Prefixo(s) identificado da pasta do modelo para gerar as predições.")
    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    identifications = args.identification
    if identifications is None:
        identifications = os.listdir(config.OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH)

    # Filtra os modelos que usam a quantificação especificada
    identifications = [identification for identification in identifications if str(args.quantification) in identification]

    print(f'Num. experimentos encontrados: {len(identifications)}')


    normalization_layer = get_normalization_layer(args.normalization, args.quantification)    
    for model_name in args.model:
        for fold in args.fold:

            df_fold = read_fold_file(fold)
            _, _, df_test = filter_train_validation_test_dataframe(df_fold)
            

            for identification in identifications:
                print(model_name, fold)
                model_path = os.path.join(config.OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH, identification, model_name, str(fold), 'model.keras')

                # Limpa a sessão do tensorflow a cada fold (just in case)
                tf.keras.backend.clear_session()

                print(f'Loading {model_name} - Fold: {fold} - {identification}')
                model = tf.keras.models.load_model(model_path)

                output_tif_dir = os.path.join(config.OUTPUT_PREDICTIONS_TRANSFER_LEARNING_PATH, 'tif',  identification, f'{model_name}', str(fold))
                os.makedirs(output_tif_dir, exist_ok=True)
                
                output_png_dir = os.path.join(config.OUTPUT_PREDICTIONS_TRANSFER_LEARNING_PATH, 'png',  identification, f'{model_name}', str(fold))
                if args.png:
                    os.makedirs(output_png_dir, exist_ok=True)

                for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
                    image_path = row['images_paths']
                    
                    patch_name = os.path.basename(image_path)
                    png_name = decode_png_patch_name(model_name, identification, patch_name)
                    
                    output_tif_file = os.path.join(output_tif_dir, patch_name)
                    output_png_file = os.path.join(output_png_dir, png_name)

                    if os.path.exists(output_tif_file) and (not args.png or os.path.exists(output_png_file) ):
                        continue

                    # patch_name = os.path.basename(image_path)
                    # patch_name = patch_name.replace('.tif', '.png')

                    with rasterio.open(image_path) as src:
                        if src.meta['count'] == 3:
                            img = src.read().transpose((1,2,0))
                        else:
                            img = src.read((6,5,4)).transpose((1,2,0))
                        meta = src.meta


                    img = normalization_layer(img).numpy()
                    img = np.asarray([img])

                    model_input = (img,)
                    prediction = model.predict(model_input, batch_size=1, verbose=0)[0]
                    prediction = np.array(prediction > 0.5, dtype=np.uint8)
                    prediction = np.squeeze(prediction)

                    meta.update(count=1)
                    meta.update(nodata=None)
                    meta.update(dtype=rasterio.uint8)
                    with rasterio.open(output_tif_file, 'w+', **meta) as dst:
                        dst.write_band(1, prediction)


                    if args.png:

                        prediction = Image.fromarray(np.asarray(prediction*255, dtype=np.uint8))
                        prediction.save(output_png_file)

    print('Done!')