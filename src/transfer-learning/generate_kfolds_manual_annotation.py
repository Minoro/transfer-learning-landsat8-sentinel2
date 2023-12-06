import os
import sys
import numpy as np

from glob import glob
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import random
from tqdm import tqdm
import rasterio

from sklearn.utils import shuffle


IMAGES_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/256/imgs'
MASKS_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/256/mask1'



# Número de imagens com mais fogo que serão usadas para treino
NUM_TOP_FIRE_IMAGES_TO_TRAIN = 5

# Tamanho do conjunto de validação
VALIDATION_FRACTION = 0.2

# Proporção de patches sem fogo para selecionar
NON_FIRE_PROPORTION = 1

# Modo de construção do conjunto de teste. 
# Quando 'n-1' utiliza as imagens não utilizadas no treino para teste
# Quando 'fixed' as imagens de treino não são utilizadas para teste. O conjunto de teste é fixo para todos os treinos.
MODE_TEST_SET = 'fixed'

# CSV com a informação do número de pixels de fogo de cada patch. Se não existir será criado
CSV_WITH_NUMBER_OF_FIRE_PIXELS_OF_EACH_PATCH = '../../resources/sentinel/masks_fire_pixel.csv'

# CSV de saida com os patches de cada FOLD
OUTPUT_PATH = f'../../resources/sentinel/'
OUTPUT_FILE = f'manual_annotation_{NUM_TOP_FIRE_IMAGES_TO_TRAIN}folds_patches_8020_mask1.csv'

RANDOM_SEED = 42

def write_csv_with_number_of_fire_pixel_of_each_patch():
    """Cria um CSV com as informação no número de pixels de fogo de cada patch.
    Também computa o valor máximo das bandas SWIR1, SWIR2 e NIR.
    """

    data = []
    images = glob(os.path.join(IMAGES_DIR, '*.tif'))

    for image in tqdm(images):
        image_name = os.path.basename(image)

        mask1 = image_name.replace('_stack', '_stack_maskf')


        with rasterio.open(image) as src:
            max_val_swir_nir = src.read((6,5,4)).max()

        with rasterio.open(os.path.join(MASKS_DIR, mask1)) as src:
            num_fire_pixels_mask1 = (src.read() != 0).sum()

        data.append({
            'sentinel_image': '_'.join(image_name.split('_')[:2]), 
            'image': image_name,
            'mask1': mask1,
            'max_val_swir_nir': max_val_swir_nir,
            'num_fire_pixels_mask1': num_fire_pixels_mask1,
        })

    df = pd.DataFrame(data)
    df.to_csv(CSV_WITH_NUMBER_OF_FIRE_PIXELS_OF_EACH_PATCH)


def make_dataframe_with_samples(df, mask_level, num_train_images=1):
    """Separa as amostras em conjuntos de treino/validação e teste.
    As amostras de treino são selecionadas entre as imagens com mais patches de fogo.
    Cada fold será composto de uma imagem de treino, as demais imagens (incluindo as não selecionadas para treino)
    serão utilizadas no conjunto de teste.
    Os patches de treino serão divididos em treino e validação.
    Para balancear o número de patches com fogo serão selecionados patches sem fogo.
    O conjunto de teste utiliza todos os patches da imagem.
    Para ordenar o número de patches é preciso informar o nível da máscara (num_fire_pixels_mask1 ou num_fire_pixels_mask2).
    """ 

    df['patch_sufix'] = df['mask1'].apply(lambda x : str(x).split('_')[-1])
    # Agrupa para verificar quantos patches de fogo cada imagem tem
    df_tmp = df[ df[mask_level] != 0 ].groupby('sentinel_image').count().reset_index()
    # Ordena as imagens por número de patches com fogo e pega as top NUM_TOP_FIRE_IMAGES_TO_TRAIN
    train_images_names = df_tmp.sort_values(mask_level).tail(NUM_TOP_FIRE_IMAGES_TO_TRAIN)['sentinel_image'].values
    
    # Agrupa as imagens para verificar o total de número de pixels
    # df_tmp = df[ df[mask_level] != 0 ].groupby('sentinel_image').sum(mask_level).reset_index()
    # train_images = df_tmp.sort_values(mask_level).tail(NUM_TOP_FIRE_IMAGES_TO_TRAIN)['sentinel_image'].values

    train_images = combinations(train_images_names, num_train_images)
    train_images = list(train_images)
    random.Random(RANDOM_SEED).shuffle(train_images)

    folds = []
    for k, train_image in enumerate(train_images, start=1):


        # Seleciona as amostras com fogo
        df_tmp = df[ (df['sentinel_image'].isin(train_image)) & (df[mask_level] != 0)]
        df_train = df_tmp.sample(frac=1-VALIDATION_FRACTION, random_state=RANDOM_SEED)
        df_validation = df_tmp.drop(df_train.index)
        # print(train_image, len(df_tmp), len(df_train), len(df_validation))

        # Adiciona amostras sem fogo (desconsidera os patchs de NODATA -> max_val_swir_nir == 0)
        df_tmp = df[ (df['sentinel_image'].isin(train_image)) & (df[mask_level] == 0) & (df['max_val_swir_nir'] !=0)]
        df_train_nofire = df_tmp.sample(len(df_train) * NON_FIRE_PROPORTION, random_state=RANDOM_SEED)
        df_validation_nofire = df_tmp[ ~df_tmp['mask1'].isin( df_train_nofire['mask1'] )].sample(len(df_validation) * NON_FIRE_PROPORTION, random_state=RANDOM_SEED)


        df_train = pd.concat((df_train_nofire, df_train), axis=0)
        df_validation = pd.concat((df_validation_nofire, df_validation), axis=0)


        df_train['set'] = 'train'
        df_validation['set'] = 'validation'

        if MODE_TEST_SET == 'fixed':
            # Usa as imagens não utilizadas no treino/validação para teste
            df_test = df[ ~df['sentinel_image'].isin(train_images_names) ].copy()
            df_test['set'] = 'test'
        else: 
            # Usa todas as imagens, com exceção da selecionada para treino, para testar a imagem
            df_test = df[ df['sentinel_image'] != train_image ].copy()
            df_test['set'] = 'test'

        df_fold = pd.concat((df_train, df_validation, df_test), axis=0)
        df_fold['fold'] = k
        df_fold['train_image'] = ','.join(train_image)
        df_fold = shuffle(df_fold, random_state=RANDOM_SEED)

        folds.append(df_fold)

    df_fold = pd.concat(folds, axis=0)

    return df_fold


if __name__ == '__main__':

    if not os.path.exists(CSV_WITH_NUMBER_OF_FIRE_PIXELS_OF_EACH_PATCH):
        write_csv_with_number_of_fire_pixel_of_each_patch()

    df = pd.read_csv(CSV_WITH_NUMBER_OF_FIRE_PIXELS_OF_EACH_PATCH)    

    df_fold = make_dataframe_with_samples(df, 'num_fire_pixels_mask1')
    df_fold.to_csv(os.path.join(OUTPUT_PATH, OUTPUT_FILE))


    print('Done!')
