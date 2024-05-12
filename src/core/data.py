import tensorflow as tf
import rasterio
import numpy as np
import pandas as pd
import os

# Max. pixel value, used to normalize the Landsat-8 images
MAX_LANDSAT_PIXEL_VALUE = 65535 

# Default non-fire mask for patches marked as "seamline" 
SEAMLINE_MASK = tf.zeros((256,256, 1)).numpy()
RANDOM_SEED = 42

g = tf.random.Generator.from_seed(RANDOM_SEED)

class LandsatDatasetBuilder():

    def __init__(self):
        self.dataset = None
        self.num_images = None
        self.batch_size = 1
        self.normalization_layer = None
        self.bands = (7,6,5)

    def use_dataframe_in_path(self, dataframe_path):
        self.dataframe_path = dataframe_path
        return self

    def use_mask(self, mask_name : str):
        self.mask_name = mask_name
        return self
    
    def use_images_path(self, images_path : str):
        self.images_path = images_path
        return self
    
    def use_masks_path(self, masks_path: str):
        self.masks_path = masks_path
        return self
    
    def use_normalization_layer(self, normalization_layer):
        self.normalization_layer = normalization_layer
        return self
    
    def use_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self
    
    def use_normalization_layer(self, normalization_layer):
        self.normalization_layer = normalization_layer
        return self
    
    def use_bands(self, bands):
        self.bands = bands
        return self

    def prepare_dataset(self, set_name):
        
        x = pd.read_csv(os.path.join(self.dataframe_path, self.mask_name, f'images_{set_name}.csv'))
        y = pd.read_csv(os.path.join(self.dataframe_path, self.mask_name, f'masks_{set_name}.csv'))
        images_paths = [ os.path.join(self.images_path, image) for image in x['images'] ]
        masks_paths = [ os.path.join(self.masks_path, mask) for mask in y['masks'] ]

        self.num_images = len(images_paths)
        self.num_masks = len(masks_paths)

        assert self.num_images == self.num_masks, "The number of images and masks don't match"

        shuffle = True
        repeat = True
        use_data_augmentation=True

        if set_name != 'train':
            shuffle = False
            repeat = False
            use_data_augmentation=False
    
        self.dataset = get_dataset_from_slices([*zip(images_paths, masks_paths)], shuffle=shuffle, buffer_size=self.num_images)
        self.dataset = map_dataset_to_open_bands_from_paths(self.dataset, self.bands)
        self.dataset = prepare_dataset(self.dataset, self.batch_size, self.normalization_layer, use_data_augmentation, repeat)

        return self
    
    def get_dataset(self):
        if self.dataset is None:
            raise Exception("Dataset is not preprared. Call the methos starting with 'use_' to set the dataset definition then call 'prepare_dataset'. ")

        return self.dataset
    
    def get_num_images(self):
        if self.dataset is None:
            raise Exception("Dataset is not preprared. Call the methos starting with 'use_' to set the dataset definition then call 'prepare_dataset'. ")

        return self.num_images


def get_landsat_images_dataset_and_num_images_from_config_and_args(config, args, set_name : str):
    dataset_builder = get_dataset_builder_from_config_and_args(config, args)

    if config.QUANTIFICATION_VALUE is not None:
        dataset_builder = dataset_builder.use_normalization_layer(tf.keras.layers.Rescaling(1./config.QUANTIFICATION_VALUE))

    ds = dataset_builder.prepare_dataset(set_name).get_dataset()

    return ds, dataset_builder.get_num_images()



def get_dataset_builder_from_config_and_args(config, args):
    builder = LandsatDatasetBuilder() \
                .use_dataframe_in_path(config.IMAGES_DATAFRAMES_PATH) \
                .use_images_path(config.IMAGES_PATH) \
                .use_masks_path(config.MASKS_PATH) \
                .use_mask(args.mask) \
                .use_bands(config.BANDS) \
                .use_batch_size(args.batch_size)
    
    return builder



def get_dataset_from_slices(paths, shuffle, buffer_size=1024):

    # Converte os caminhos para um dataset, se iterado retorna uma tupla com o caminho da imagem e da máscara
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=RANDOM_SEED) 

    return dataset
    
def map_dataset_to_open_bands_from_paths(dataset, bands):
    if type(bands) == type(int):
        bands = [bands]
    bands = list(bands)

    return map_dataset_to_open_bands_from_paths_with_custom_function(dataset, open_landsat_image_and_mask, bands)

def map_dataset_to_open_bands_from_paths_with_custom_function(dataset, open_function, open_args=[]):
    return dataset.map(lambda x: tf.py_function(open_function, [x, open_args], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)


def get_landsat_dataset_from_paths(image_paths, masks_paths, batch_size=8, normalization_layer=None, use_data_augmentation=False, shuffle=False, repeat=True):
    # Converte os caminhos para um dataset, se iterado retorna uma tupla com o caminho da imagem e da máscara
    dataset = tf.data.Dataset.from_tensor_slices([*zip(image_paths, masks_paths)])

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=RANDOM_SEED) 

    # Passa a função para abrir a imagem e a máscara a partir da tupla com os caminhos
    dataset = dataset.map(lambda x:  tf.py_function(open_landsat_image_and_mask, [x, (7,6,5)], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    
    return prepare_dataset(dataset, batch_size, normalization_layer, use_data_augmentation, repeat)


def get_sentinel_dataset_from_paths(image_paths, masks_paths, batch_size=8, normalization_layer=None, use_data_augmentation=False, shuffle=False, repeat=True):
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
    dataset = dataset.map(lambda x:  tf.py_function(open_sentinel_image_and_mask, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    
    return prepare_dataset(dataset, batch_size=batch_size, normalization_layer=normalization_layer, use_data_augmentation=use_data_augmentation, repeat=repeat)
    


def prepare_dataset(dataset, batch_size=8, normalization_layer=None, use_data_augmentation=False, repeat=True):
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

    dataset = dataset.batch(batch_size) 
    
    if use_data_augmentation:
        dataset = dataset.map(data_augmentation)
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset



def open_sentinel_image_and_mask(paths):
    """Abre uma imagem e a máscara a partir do tensor contendo os caminhos
    A primeira posição do tensor deve conter o caminho para a imagem
    A segunda posição do tensor deve conter o caminho para a máscra
    """
    # print(paths)
    image_path = paths[0].numpy().decode()
    mask_path = paths[1].numpy().decode()

    # Lê as bandas SWIR-2, SWIR-1 e NIR, nessa ordem
    # O transpose é aplicado para converter para channels-last
    with rasterio.open(image_path) as src:
        if src.meta['count'] == 3:
            img = src.read().transpose((1,2,0))
        else:
            img = src.read((6,5,4)).transpose((1,2,0))

    # Lê a máscara com a dimensão do canal (256,256,1)
    if mask_path == '':
        return img, SEAMLINE_MASK
    
    with rasterio.open(mask_path) as src:
        mask = src.read().transpose((1,2,0))
        # mask = src.read(1)
    
    return img, mask



def data_augmentation(input_image, input_mask):

    rand = g.uniform([1])
    # tf.print('Rand 1:', rand)
    if rand > 0.5:
        # tf.print('Flip horizontal')
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
        
    rand = g.uniform([1])
    # tf.print('Rand 2:', rand)
    if rand > 0.5:
        # tf.print('Flip vertical')
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    return input_image, input_mask



def get_img_765bands(path):
    img = get_img_bands(path, (7,6,5)) 
    img = np.float32(img)
   
    return img


def get_img_bands(path, bands):
    with rasterio.open(path) as src:
        img = src.read(bands)
    
    img = img.transpose((1, 2, 0))    
    img = np.float32(img)

    return img


def get_mask_arr(path):
    # img = rasterio.open(path).read()
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg


def open_landsat_image_and_mask(paths, bands):

    image_path = paths[0].numpy().decode()
    mask_path = paths[1].numpy().decode()
    
    # print(bands.numpy())
    return get_img_bands(image_path, tuple(bands.numpy())), get_mask_arr(mask_path)




