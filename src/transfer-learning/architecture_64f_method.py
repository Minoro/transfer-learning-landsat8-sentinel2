import sys
import os
import warnings

# define the GPU before import TF just in case
CUDA_DEVIDE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVIDE)
sys.path.append('../')
warnings.filterwarnings("ignore")

from glob import glob
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Custom Scripts
from core.unet import get_unet, freeze_unet_layers
from core.data import data_augmentation, open_image_and_mask
from core.metrics import *

# Threshold method to fine-tune the network
# Accept: Liu, Kato-Nakamura, Murphy, Voting and Intersection
METHOD = 'Liu'

MASK_LEVEL = 'mask1'
NFILTERS = 64


# Name to identify the saved model, it will be the "prefix" of the final name
MODEL_NAME = f'5fold_8020_{MASK_LEVEL}_unet{NFILTERS}f'

# File with the images of each fold to evaluate the fine-tune of the model
FOLDS_FILE = f'../../resources/sentinel/manual_annotation_5folds_patches_8020_mask1.csv'

# Base models available. It accept the following values:
# 'Kumar-Roy', 'Schroeder', 'Murphy', 'Voting', 'Intersection' 
PRE_TRAINED_MASKS = [
    'Kumar-Roy', 'Schroeder', 'Murphy', 'Voting', 'Intersection',
]

# Path to the pre-traeined weights, trained with Landsat-8 images
PRE_TRAINED_WEIGHTS_DIR = '../../resources/landsat/weights'

# Basic network configuration
LOSS_FUNCTION='bce'
EPOCHS = 20
BATCH_SIZE = 8
LR=1e-4
IMAGE_SHAPE = (256,256,3)
RANDOM_SEED = 42


# Normalization strategy. Accept
# "fixed" : First normalize by 10000, then apply a batch normalization layer
# "no-bn" : Just normalize by QUANTIFICATION_VALUE, the BN layer will not be used
# None: Don't normalize
NORMALIZATION_MODE = 'fixed' 
# Valor de quantificação do Sentinel
QUANTIFICATION_VALUE = 10000.0

# Early stop configuration
USE_DATA_AUGMENTATION = True
EARLY_STOP_PATIENCE = 5 
EARLY_STOP_RESTORE_BEST = True


IMAGES_DIR = '../../resources/sentinel/Sentinel2/manual_annotated/256/imgs'
MASKS_DIR = f'../../resources/sentinel/Sentinel2/manual_annotated/256/methods/{METHOD}'

# Transfer-Learning strategy. Accept the following values
# "freeze_all" - Frozen weights (no learning); the BN layer is free to learn
# "unfreeze" - Unfrozen wights (update all weights)
# "freeze_encoder" - Frozen-Encoder (the first half is frozen and the second half is free to learn).
# "random" - Will not load the pre-trained weights. All layers are free to learn.
CONFIGS = [
    'freeze_all', 'unfreeze', 'freeze_encoder',
]

# Define the final name of the model
output_folder_name = f'{MODEL_NAME}_{LOSS_FUNCTION}'
if USE_DATA_AUGMENTATION:
    output_folder_name += '_augmentation'
else:
    output_folder_name += '_noaugmentation'

# Path to save the final results
OUTPUT_DIR = os.path.join('../../resources/transfer_learning/output', 'results', output_folder_name)
# Path to save the final models (fine-tuned)
WEIGHTS_DIR = os.path.join('../../resources/transfer_learning/weights', output_folder_name)

# Folds that will be used by the script
FOLDS = [1, 2, 3, 4, 5]


def get_dataset_from_paths(image_paths, masks_paths, normalization_layer=None, use_data_augmentation=False, shuffle=False, repeat=True):
    """ Returns a tf.data.Dataset to feed the networks.
    It can be used to iterate over the dataset, it generates the batch of images and masks.
    Returns a tensor with the shape [[BATCH_SIZE, 256, 256, 3], [BATCH_SIZE, 256, 256, 1]].
    """

    # Adjust the paths to be used by the dataset
    dataset = tf.data.Dataset.from_tensor_slices([*zip(image_paths, masks_paths)])
    if shuffle:
        # Shuffle the paths instead of the "open" images to save memory
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=RANDOM_SEED) 

    # Give the paths of the images and masks to the function "open_image_and_mask" which return the tensor representing the images and masks
    dataset = dataset.map(lambda x:  tf.py_function(open_image_and_mask, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    
    return prepare_dataset(dataset, normalization_layer, use_data_augmentation, repeat)
    

def prepare_dataset(dataset, normalization_layer=None, use_data_augmentation=False, repeat=True):
    """Apply the transformations over the tf.data.Dataset.
    It returns dataset adjusted to generate batchs.
    The normalization_layer will be used to transform the images to a new base scale
    The final dataset will be returned and can be used to iterate over the batchs of images and masks
    """

    if normalization_layer is not None:
        dataset = dataset.map(lambda x,y: (normalization_layer(x), y))
                
    # dataset = dataset.cache()
    if repeat:
        # repete o dataset para formar os batchs do mesmo tamanho
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE) 
    
    if use_data_augmentation:
        dataset = dataset.map(data_augmentation)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    df = pd.read_csv(FOLDS_FILE)


    df['images_paths'] = df['image'].apply(lambda x : os.path.join(IMAGES_DIR, x))
    df['masks_paths'] = df[f'{MASK_LEVEL}'].apply(lambda x : os.path.join(MASKS_DIR, x))



    for k in FOLDS:
        print(f'Fold: {k}')

        df_train = df[ (df['fold'] == k) & (df['set'] == 'train')]
        df_validation = df[ (df['fold'] == k) & (df['set'] == 'validation')]
        df_test = df[ (df['fold'] == k) & (df['set'] == 'test')]

        train_images_paths = df_train['images_paths'].values
        train_masks_paths = df_train['masks_paths'].values
        
        validation_images_paths = df_validation['images_paths'].values
        validation_masks_paths = df_validation['masks_paths'].values

        test_images_paths = df_test['images_paths'].values
        test_masks_paths = df_test['masks_paths'].values


        # Normalize the input data
        normalization_layer = None
        if NORMALIZATION_MODE == 'fixed' or NORMALIZATION_MODE == 'no-bn':
            # Sentinel-2 images are quantified by 10,000
            normalization_layer = tf.keras.layers.Rescaling(1./QUANTIFICATION_VALUE)
        elif NORMALIZATION_MODE is not None:
            raise 'Invalid Normalization Method'

        for pre_treined_mask in PRE_TRAINED_MASKS:
            
            models_path = os.path.join(PRE_TRAINED_WEIGHTS_DIR, pre_treined_mask.lower(), f'unet_{NFILTERS}f_2conv_765', f'model_unet_{pre_treined_mask}_765_final_weights.h5')
            output_dir = os.path.join(OUTPUT_DIR, METHOD, str(k), pre_treined_mask)
            os.makedirs(output_dir, exist_ok=True)

            for config in CONFIGS:
                if os.path.exists(os.path.join(output_dir, f"{config}_results.json")):
                    print(f'Ignoring {METHOD} - {config} - Fold: {k} - Pretrained: {pre_treined_mask}')
                    continue
                
                tf.keras.backend.clear_session()

                unet = get_unet(n_channels=3, n_filters=NFILTERS)

                if config != 'random':
                    # Carrega os pesos e congela as camadas
                    unet.load_weights(models_path)
                    unet = freeze_unet_layers(unet, config)

                model = unet
                if NORMALIZATION_MODE == 'fixed':
                    # Adiciona uma camada BN logo no inicio da rede para normalizar a entrada
                    model = tf.keras.Sequential(
                        [
                            tf.keras.Input(shape=IMAGE_SHAPE), 
                            tf.keras.layers.BatchNormalization(),
                            unet
                        ]
                    )

                metrics = {
                    'dice_coef': dice_coef,
                    'P': tf.keras.metrics.Precision(),
                    'R': tf.keras.metrics.Recall(),
                }

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(LR),
                    loss=LOSS_FUNCTION,
                    metrics=[metrics.values()],
                )
                # model.summary()
        
                # Create the generators to feed the network 
                train_dataset = get_dataset_from_paths(train_images_paths, train_masks_paths, normalization_layer=normalization_layer, use_data_augmentation=USE_DATA_AUGMENTATION, shuffle=True)
                val_dataset = get_dataset_from_paths(validation_images_paths, validation_masks_paths, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False)
                test_dataset = get_dataset_from_paths(test_images_paths, test_masks_paths, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False, repeat=False)
              
                # Fine-tune the models. Even the "freeze_all" configuration to adjust the BN layer
                print(f'Modelo: {config} - Fold: {k} - Pesos: {pre_treined_mask}')
                es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=EARLY_STOP_RESTORE_BEST, verbose=1)
                history = model.fit(
                    train_dataset, 
                    validation_data=val_dataset, 
                    steps_per_epoch=len(train_images_paths) // BATCH_SIZE,
                    validation_steps=len(validation_images_paths) // BATCH_SIZE,
                    epochs=EPOCHS, 
                    callbacks=[es]
                )

                # Save the history file with the loss value for each epoch
                out_file = open(os.path.join(output_dir, f"{config}_history.json"), "w")
                json.dump(history.history, out_file, default=str)
                out_file.close()
                del history

                # Save the final model to be loaded in the future
                model.save(os.path.join(WEIGHTS_DIR, f'{pre_treined_mask}_fold_{k}_nsamples_{config}_{LOSS_FUNCTION}'))

                # Release the memory
                del train_dataset
                del val_dataset

                # Evalute the model with the test set and save the results
                print(f'Evaluating Model: {METHOD} - {config} - Fold: {k} - Pre-trained: {pre_treined_mask}')
                results = evaluate_dataset(model, test_dataset)
                print(results)
                out_file = open(os.path.join(output_dir, f"{config}_results.json"), "w")
                json.dump(results, out_file)
                out_file.close()
                del results

                del model
                del unet
            
                print('Training Done!')
                        
    print('Done!')
    