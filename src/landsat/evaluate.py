import sys
import os
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

import argparse
from datetime import datetime

sys.path.append('../')
import landsat_config as config
from core.data import get_landsat_images_dataset_and_num_images_from_config_and_args
from core.models import get_model, get_models_available
from core.metrics import evaluate_dataset

from utils.report import Reporter



os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_DEVICE)

TEST_SET_NAME = 'test'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network with Landsat-8 images, using the masks informed in the arguments.')
    parser.add_argument('--model', action="store", choices=get_models_available(), default=config.MODEL, help="Model to train")
    parser.add_argument('--mask', action="store", choices=os.listdir(config.IMAGES_DATAFRAMES_PATH), default=config.MASK, help="Mask to train the base model")
    parser.add_argument('--bands', action='store', default=config.BANDS, type=tuple, help="Bands to train the model")
    parser.add_argument('--batch-size', action="store", default=config.BATCH_SIZE, type=int, help="Batch size for training")

    args = parser.parse_args()

    test_ds, num_test_images = get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, TEST_SET_NAME)

    print(f'Num. Test images: {num_test_images}')
    model = get_model(args.model, input_shape=config.IMAGE_SHAPE, num_classes=1)
    results = evaluate_dataset(model, test_ds)

    reporter = Reporter()
    reporter.add(results)
    reporter.add(config)
    reporter.add(args)
    output_dir = os.path.join(config.LANDSAT_OUTPUT_DIR, args.model, args.mask)
    out_file = open(os.path.join(output_dir, f"results_{args.model}_{args.mask}.json"), "w")
    reporter.to_json(out_file)
    
    print(results)
    
    print('Done!')