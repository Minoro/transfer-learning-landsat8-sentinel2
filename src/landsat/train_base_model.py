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
from core.data import get_landsat_images_dataset_and_num_images_from_config_and_args
from core.models import get_model, get_models_available
from utils.report import Reporter

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_DEVICE)

TRAIN_SET_NAME = 'train'
VALIDATION_SET_NAME = 'val'

def train_model(args):

    output_dir = os.path.join(config.LANDSAT_OUTPUT_DIR, args.model, args.mask)
    os.makedirs(output_dir, exist_ok=True)

    train_ds, num_train_images = get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, TRAIN_SET_NAME)
    val_ds, num_val_images = get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, VALIDATION_SET_NAME)

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
    model_weights_output = os.path.join(output_dir, config.FINAL_WEIGHTS_OUTPUT)
    model.save_weights(model_weights_output)
    print("Weights Saved: {}".format(model_weights_output))

    print('Saving history...')
    out_file = open(os.path.join(output_dir, f"history_{args.model}_{args.mask}.json"), "w")
    json.dump(history.history, out_file, default=str)
    out_file.close()
    print("History Saved!")



    # Save the train parameters
    print("Saving training parameters")
    reporter = Reporter()
    reporter.add(config)
    reporter.add(args)
    reporter.push('num_train_images', num_train_images)
    reporter.push('num_val_images', num_val_images)
    reporter.push('started_at', started_at)
    reporter.push('finished_at', finished_at)
    reporter.push('history', history.history)
    out_file = open(os.path.join(output_dir, f"parameters_{args.model}_{args.mask}.json"), "w")
    reporter.to_json(out_file)
    print('Parameters saved')
    
    print('Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a network with Landsat-8 images, using the masks informed in the arguments.')
    parser.add_argument('--model', action="store", choices=get_models_available(), default=config.MODEL, help="Model to train")
    parser.add_argument('--mask', action="store", choices=os.listdir(config.IMAGES_DATAFRAMES_PATH), default=config.MASK, help="Mask to train the base model")
    parser.add_argument('--batch-size', action="store", default=config.BATCH_SIZE, type=int, help="Batch size for training")
    parser.add_argument('--lr', action="store", default=config.LR, type=float, help="Learning Rate for training")
    parser.add_argument('--epochs', action="store", default=config.EPOCHS, type=int, help="Number of epochs for training")
    parser.add_argument('--early-stoping-patience', action="store", default=config.EARLY_STOP_PATIENCE, type=int, help="Early Stoping: number of epochs to waiting until stop training if the model do not improve")
    parser.add_argument('--bands', action='store', default=config.BANDS, type=tuple, help="Bands to train the model")

    args = parser.parse_args()

    train_model(args)

    print('Done!')