
EPOCHS = 50
BATCH_SIZE = 8
LR = 0.001
IMAGE_SHAPE = (256, 256, 3)

IMAGES_DATAFRAMES_PATH = '../../resources/landsat/dataframes/'

IMAGES_PATH = '../../resources/landsat/images/patches'
MASKS_PATH = '../../resources/landsat/images/masks'

OUTPUT_DIR = '../../resources/landsat/output/weights'

EARLY_STOP_PATIENCE = 5 
CHECKPOINT_PERIOD = 'epoch'
CHECKPOINT_MODEL_NAME = 'checkpoint-epoch_{}_{}_{{epoch:02d}}.weights.h5'

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_765_final_weights.h5'

