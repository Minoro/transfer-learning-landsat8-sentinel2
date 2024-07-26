SENTINEL_MANUAL_ANNOTATIONS_IMAGES_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/imgs_256'
SENTINEL_MANUAL_ANNOTATIONS_MASK_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/annotations'
SENTINEL_MANUAL_ANNOTATIONS_METHODS_PATH = '../../resources/sentinel/Sentinel2/manual_annotated/gdal_croped/methods'


SENTINEL_IDENTIFICATION_MANUAL_ANNOTATION_MASK = '_20m_stack_maskf_'
SENTINEL_IDENTIFICATION_MANUAL_ANNOTATION_METHOD_PREDICTION = '_mask_'

SENTINEL_OUTPUT_RESULTS_METHODS_PATH = '../../resources/sentinel/output/results/methods'

################### FOLDS ########################################################

OVERRIDE_CSV_NUM_FIRE_PIXELS_PER_PATCH = True

SENTINEL_MANUAL_ANNOTATIONS_FOLDS_CSV_PATH = '../../resources/sentinel/dataframes/manual_annotation_5folds_patches_8020_mask1.csv'

CSV_NUM_FIRE_PIXELS_PER_PATCH_PATH = '../../resources/sentinel/dataframes/num_fire_pixels_per_patch.csv'

GENERATE_VALIDATION_FOLD = True

NUM_FOLDS = 5

STRATIFIED_FOLDS = False

################################## TRANSFER LEARNING ################################

PRETRAINED_MODELS = ('unet', )

MODEL = 'unet'

NORMALIZATION_MODE = 'bn'

FINE_TUNING = True

TRANSFER_LEARNING_STRATEGY = ('freeze_encoder', 'unfreeze', 'freeze_all')

IDENTIFICATION_PREFIX = ''

BASE_MODEL = ''

OUTPUT_RESULTS_TRANSFER_LEARNING_PATH = '../../resources/sentinel/output/results/transfer_learning'

PRETRAINED_WEIGHTS_PATH = '../../resources/landsat/output'

EPOCHS = 20

BATCH_SIZE = 8

LR=1e-4

LOSS_FUNCTION='bce'

USE_DATA_AUGMENTATION = True

EARLY_STOP_PATIENCE = 5

EARLY_STOP_RESTORE_BEST = True

OUTPUT_WEIGHTS_TRANSFER_LEARNING_PATH = '../../resources/sentinel/output/weights'

OUTPUT_PREDICTIONS_TRANSFER_LEARNING = '../../resources/sentinel/output/prediction'

CHECKPOINT_PERIOD = 'epoch'
CHECKPOINT_MODEL_NAME = 'checkpoint-epoch_{}_{}_{{epoch:02d}}.keras'

OUTPUT_GRADCAM_DIR = '../../resources/sentinel/output/gradcam'


######################### COMMONS #######################################
CUDA_DEVICE = 0

RANDOM_SEED = 42

# Landsat bands
BANDS = (7,6,5)

IMAGE_SHAPE = (256,256,3)

THRESHOLDING_METHODS = (
    'KatoNakamura', 'Liu', 'Murphy'
)

PRETRAINED_MASKS = (
    'Intersection', 'Kumar-Roy', 'Murphy', 'Schroeder', 'Voting'
)

SENTINEL_QUANTIFICATION_VALUE = 10000.0


