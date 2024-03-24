import sys
import os
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import argparse
from datetime import datetime

sys.path.append('../')
import landsat_config as config
from core.data import get_landsat_dataset_from_paths
from core.models import get_model, get_models_available

TRAIN_SET_NAME = 'train'
VALIDATION_SET_NAME = 'val'
TEST_SET_NAME = 'test'


def train_model(args):
    
    output_dir = os.path.join(config.OUTPUT_DIR, args.model, args.mask)
    os.makedirs(output_dir, exist_ok=True)

    train_ds, num_train_images = get_images_dataset_and_num_images(args.mask, TRAIN_SET_NAME, args.batch_size)
    val_ds, num_val_images = get_images_dataset_and_num_images(args.mask, VALIDATION_SET_NAME, args.batch_size)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():

        model = get_model(args.model, input_shape=config.IMAGE_SHAPE, num_classes=1)
        metrics = {
            'P': tf.keras.metrics.Precision(),
            'R': tf.keras.metrics.Recall(),
        }

        model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss = 'binary_crossentropy', metrics=metrics.values())
        model.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.early_stoping_patience, restore_best_weights=True)
    checkpoint_name = os.path.join(output_dir, config.CHECKPOINT_MODEL_NAME.format(args.model, args.mask))
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='auto', save_freq=config.CHECKPOINT_PERIOD)

    print('Training using {}...'.format(args.mask))
    started_at = datetime.now()
    history = model.fit(
        train_ds,
        steps_per_epoch=num_train_images // args.batch_size,
        validation_data=val_ds,
        validation_steps=num_val_images // args.batch_size,
        callbacks=[checkpoint, es],
        # callbacks=[es],
        epochs=args.epochs
    )
    finished_at = datetime.now()
    print('Train finished!')

    print('Saving weights')
    model_weights_output = os.path.join(output_dir, config.FINAL_WEIGHTS_OUTPUT.format(args.model, args.mask))
    model.save_weights(model_weights_output)
    print("Weights Saved: {}".format(model_weights_output))

    print('Saving history...')
    out_file = open(os.path.join(output_dir, f"history_{args.model}_{args.mask}.json"), "w")
    json.dump(history.history, out_file, default=str)
    out_file.close()
    print("History Saved!")


    # Save the train parameters
    print("Saving training parameters")
    parameters = vars(args)
    parameters['num_train_images'] = num_train_images
    parameters['num_val_images'] = num_val_images
    parameters['started_at'] = started_at
    parameters['finished_at'] = finished_at
    out_file = open(os.path.join(output_dir, f"parameters_{args.model}_{args.mask}.json"), "w")
    json.dump(parameters, out_file, default=str)
    out_file.close()
    print('Parameters saved')
    
    print('Done!')

def get_images_dataset_and_num_images(mask_name : str, set_name : str, batch_size : int):
    assert mask_name in os.listdir(config.IMAGES_DATAFRAMES_PATH)
    assert set_name in ['train', 'val', 'test']

    x = pd.read_csv(os.path.join(config.IMAGES_DATAFRAMES_PATH, mask_name, f'images_{set_name}.csv'))
    y = pd.read_csv(os.path.join(config.IMAGES_DATAFRAMES_PATH, mask_name, f'masks_{set_name}.csv'))


    images_paths = [ os.path.join(config.IMAGES_PATH, image) for image in x['images'] ]
    masks_paths = [ os.path.join(config.MASKS_PATH, mask) for mask in y['masks'] ]

    shuffle = True
    repeat = True

    if set_name != 'train':
        shuffle = False
        repeat = False

    ds = get_landsat_dataset_from_paths(images_paths, masks_paths, batch_size=batch_size, use_data_augmentation=False, shuffle=shuffle, repeat=repeat)

    return ds, len(images_paths)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a network with Landsat-8 images, using the masks informed in the arguments.')
    parser.add_argument('--model', action="store", choices=get_models_available(), required=True, help="Model to train")
    parser.add_argument('--mask', action="store", choices=os.listdir(config.IMAGES_DATAFRAMES_PATH), required=True, help="Mask to train the base model")
    parser.add_argument('--batch-size', action="store", default=config.BATCH_SIZE, type=int, help="Batch size for training")
    parser.add_argument('--lr', action="store", default=config.LR, type=float, help="Learning Rate for training")
    parser.add_argument('--epochs', action="store", default=config.EPOCHS, type=int, help="Number of epochs for training")
    parser.add_argument('--early-stoping-patience', action="store", default=config.EARLY_STOP_PATIENCE, type=int, help="Early Stoping: number of epochs to waiting until stop training if the model do not improve")

    args = parser.parse_args()

    train_model(args)

    print('Done!')