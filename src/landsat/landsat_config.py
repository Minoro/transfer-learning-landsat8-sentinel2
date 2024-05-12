CUDA_DEVICE = 0

EPOCHS = 50
BATCH_SIZE = 8
LR = 0.001
IMAGE_SHAPE = (256, 256, 3)
# Tupla com as bandas a serem utilizadas 
# O número da banda é referente ao indice da imagem tif
BANDS = (7,6,5)

# Max. pixel value, used to normalize the Landsat-8 images
QUANTIFICATION_VALUE = 65535 

MODEL='unet'
MASK='Voting'

IMAGES_DATAFRAMES_PATH = '../../resources/landsat/dataframes/'

IMAGES_PATH = '../../resources/landsat/images/patches'
MASKS_PATH = '../../resources/landsat/masks/voting'

LANDSAT_OUTPUT_DIR = '../../resources/landsat/output/'

EARLY_STOP_PATIENCE = 5 
CHECKPOINT_PERIOD = 'epoch'
CHECKPOINT_MODEL_NAME = 'checkpoint-epoch_{}_{}_{{epoch:02d}}.weights.h5'

FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_{}_final_weights.h5'.format(MODEL, MASK, ''.join([str(b) for b in BANDS]))


LANDSAT_MANUAL_ANNOTATIONS_MASK_PATH = '../../resources/landsat/manual_annotations/patches/manual_annotations_patches' 
LANDSAT_MANUAL_ANNOTATIONS_IMAGES_PATH = '../../resources/landsat/manual_annotations/patches/landsat_patches'

