import os
import pandas as pd
import sys
import argparse
import json

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

import config

sys.path.append('../')
from core.models import get_normalization_layer, get_model, freeze_backbone_layers, add_bn_at_start
from core.metrics import get_model_metrics, evaluate_dataset
from core.data import get_sentinel_dataset_from_paths



def read_fold_file(args, fold):    
    df_folds = pd.read_csv(args.csv_folds_file)
    return df_folds[ df_folds['fold'] == fold ]


def get_train_validation_test_dataset_using_normalization_layer(df_fold, normalization_layer, args):
    df_train, df_validation, df_test = filter_train_validation_test_dataframe(df_fold, args.mask)

    train_dataset = get_sentinel_dataset_from_paths(df_train['images_paths'], df_train['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=config.USE_DATA_AUGMENTATION, shuffle=True, landsat_bands=args.landsat_bands)
    val_dataset = get_sentinel_dataset_from_paths(df_validation['images_paths'], df_validation['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False, landsat_bands=args.landsat_bands)
    test_dataset = get_sentinel_dataset_from_paths(df_test['images_paths'], df_test['masks_paths'], batch_size=config.BATCH_SIZE, normalization_layer=normalization_layer, use_data_augmentation=False, shuffle=False, repeat=False, landsat_bands=args.landsat_bands)

    return train_dataset, val_dataset, test_dataset


def filter_train_validation_test_dataframe(df_fold, mask_type):    
    df_fold['images_paths'] = df_fold.apply(build_image_path, axis=1)

    if mask_type == 'manual-annotation':
        df_fold['masks_paths'] = df_fold.apply(build_mask_path, axis=1)
    else:
        df_fold['masks_paths'] = df_fold.apply(build_mask_method_path(mask_type), axis=1)

    df_train = df_fold[(df_fold['set'] == 'train')]
    df_validation = df_fold[(df_fold['set'] == 'validation')]
    df_test = df_fold[(df_fold['set'] == 'test')]

    return df_train, df_validation, df_test

def build_image_path(row):
    return os.path.join(config.SENTINEL_MANUAL_ANNOTATIONS_IMAGES_PATH, row['image'])
    
def build_mask_path(row):
    return os.path.join(config.SENTINEL_MANUAL_ANNOTATIONS_MASK_PATH, row['mask1'])
    
def build_mask_method_path(method):
    return lambda x : os.path.join(config.SENTINEL_MANUAL_ANNOTATIONS_METHODS_PATH, method, x['mask1'].replace('_20m_stack_maskf_', '_mask_'))



def get_pretrained_model(model_name, base_model, args):
    input_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], len(args.landsat_bands))
    model = get_model(model_name, input_shape)

    bands_names = ''.join(['B'+str(b) for b in args.landsat_bands])
    bands = ''.join([str(b) for b in args.landsat_bands])

    models_path = os.path.join(config.PRETRAINED_WEIGHTS_PATH, model_name, base_model, bands_names, f'model_{model_name}_{base_model}_{bands}_final_weights.h5')
    model.load_weights(models_path)

    return model

def define_output_folder_name(args):
    identification = args.identification
    if identification != '':
        identification += '_'

    bands_names = ''.join(['B'+str(b) for b in args.landsat_bands])
    output_folder_name = f'{identification}unet_{args.normalization}_{str(args.quantification)}_TL_with_FT_{bands_names}'
    if args.epochs == 0 or args.no_tuning:
        output_folder_name = f'{identification}unet_{args.normalization}_{str(args.quantification)}_TL_no_FT_{bands_names}'

    return output_folder_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning do Landsat-8 para o Sentinel-2.')
    
    parser.add_argument('--base-models', action="store", choices=config.PRETRAINED_MASKS, default=config.PRETRAINED_MASKS, nargs='*', help="Modelo pré-treinado com a máscara específicada para realizar fine-tuning" )

    parser.add_argument('--mask', action='store', choices=['manual-annotation', 'KatoNakamura', 'Murphy', 'Liu'], default='manual-annotation')
    parser.add_argument('--transfer-learning-strategy', action='store', choices=['unfreeze', 'freeze_encoder', 'freeze_all'], default=config.TRANSFER_LEARNING_STRATEGY, nargs='*')
    parser.add_argument('--landsat-bands', action='store', default=config.BANDS, nargs='*', help="Bands to train the model")
    parser.add_argument('--csv-folds-file', action='store', type=str, default=config.SENTINEL_MANUAL_ANNOTATIONS_FOLDS_CSV_PATH, help="CSV com a divisão dos folds")
    parser.add_argument('--fold', action="store", choices=list(range(1, config.NUM_FOLDS+1)), default=list(range(1, config.NUM_FOLDS+1)), type=int, nargs='*', help="Número(s) do(s) fold(s) para ser(em) avaliado(s)")
    parser.add_argument('--normalization', action="store", choices=['bn', 'no-bn', None], default=config.NORMALIZATION_MODE, help="Modo de normalização da imagem de entrada da rede")
    parser.add_argument('--quantification', action="store", type=float, default=config.SENTINEL_QUANTIFICATION_VALUE, help='Valor de quantificação das imagens do Sentinel-2. Se houver normalização, a imagem será dividida por esse valor')
    parser.add_argument('--epochs', action="store", type=int, default=config.EPOCHS, help="Número de épocas para transfer learning. Se zero não performar fine tuning (o mesmo que informar --no-tuning)")
    parser.add_argument('--no-tuning', action="store_true", default=(not config.FINE_TUNING), help="Se informado não performa fine tuning, o mesmo que informar o número de épocas igual a zero.")
    parser.add_argument('--checkpoint-freq', action="store", choices=['epoch']+list(range(1,20)), default=-1, help="Frequencia, em épocas, para fazer checkpoint do modelo")
    parser.add_argument('--gpu', action="store", type=str, default=str(config.CUDA_DEVICE), help="Dispositivo GPU usado")
    parser.add_argument('--identification', action="store", type=str, default=config.IDENTIFICATION_PREFIX, help="Prefixo da pasta para identificar o experimento sendo executado.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tuning = True        
    if args.epochs == 0 or args.no_tuning:
        tuning = False
        print('Fine tuning desativado!')

    model_name = 'unet'
    normalization_layer = get_normalization_layer(args.normalization, args.quantification)    
    for base_model in args.base_models:
        output_folder_name = define_output_folder_name(args)

        for transfer_learning_strategy in args.transfer_learning_strategy:
            for fold in args.fold:    
                
                output_dir = os.path.join(config.OUTPUT_RESULTS_TRANSFER_LEARNING_PATH, output_folder_name, base_model, str(fold))
                os.makedirs(output_dir, exist_ok=True)

                output_results_file = os.path.join(output_dir, f"{args.mask}_{transfer_learning_strategy}_results.json")
                if os.path.exists(output_results_file):
                    print(f'Ignorando modelo: {model_name} - Fold: {fold} - Pretrained: {base_model} - Strategy: {transfer_learning_strategy} - Mask: {args.mask}')
                    continue
                
                output_weights_dir = os.path.join(config.OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH, output_folder_name, base_model, str(fold))
                os.makedirs(output_weights_dir, exist_ok=True)
                

                # Limpa a sessão do tensor flow para carregar um novo modelo
                tf.keras.backend.clear_session()

                # Carrega o modelo e congela as camadas
                model = get_pretrained_model(model_name, base_model, args)
                print(transfer_learning_strategy)
                model = freeze_backbone_layers(model, transfer_learning_strategy)

                if args.normalization == 'bn':
                    model = add_bn_at_start(model)
                

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(config.LR),
                    loss=config.LOSS_FUNCTION,
                    metrics=[get_model_metrics()],
                )
                
                df_fold = read_fold_file(args, fold)

                df_train, df_validation, df_test = filter_train_validation_test_dataframe(df_fold, args.mask)
                train_dataset, val_dataset, test_dataset = get_train_validation_test_dataset_using_normalization_layer(df_fold, normalization_layer, args)

                if tuning:
                    print(f'Modelo: {model_name} - Fold: {fold} - Pretrained: {base_model} - Strategy: {transfer_learning_strategy} - Mask: {args.mask}')
                    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.EARLY_STOP_PATIENCE, restore_best_weights=config.EARLY_STOP_RESTORE_BEST, verbose=1)
                    callbacks = [es]

                    if args.checkpoint_freq != -1:
                        checkpoint_name = os.path.join(output_weights_dir, config.CHECKPOINT_MODEL_NAME.format(model_name, base_model))
                        # checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='auto', save_freq=args.checkpoint_freq)
                        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, mode='auto', save_freq=args.checkpoint_freq)
                        callbacks.append(checkpoint)

                    history = model.fit(
                        train_dataset, 
                        validation_data=val_dataset, 
                        steps_per_epoch=len(df_train) // config.BATCH_SIZE,
                        validation_steps=len(df_validation) // config.BATCH_SIZE,
                        epochs=args.epochs, 
                        callbacks=callbacks
                    )

                    # Save the history file with the loss value for each epoch
                    with open(os.path.join(output_dir, f"{transfer_learning_strategy}_history.json"), "w") as f:
                        json.dump(history.history, f, default=str)

                    del history

                
                model.save(os.path.join(output_weights_dir, f'model_{transfer_learning_strategy}.keras'))

                # Release the memory
                del train_dataset
                del val_dataset

                print(f'Evaluating model: {model_name} - Fold: {fold} - Pretrained: {base_model} - Mask: {args.mask}')
                results = evaluate_dataset(model, test_dataset)
                print(results)
                with open(output_results_file, "w") as f:
                    json.dump(results, f)
                del results

                del model
                print(f'Resultados salvos no arquivo: {output_results_file}')
            
        print('Done!')