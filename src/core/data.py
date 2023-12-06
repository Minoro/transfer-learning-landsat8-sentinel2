import tensorflow as tf
import rasterio

g = tf.random.Generator.from_seed(42)



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def read_tfrecord_raw_example(example):
    
    data = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask_level1': tf.io.FixedLenFeature([], tf.string),
        'mask_level2': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'metadata': tf.io.VarLenFeature(tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'tile': tf.io.FixedLenFeature([], tf.string),
        'patch': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(example, data)

    image = tf.io.parse_tensor(content['image'], out_type=tf.uint16)
    mask_level1 = tf.io.parse_tensor(content['mask_level1'], out_type=tf.uint8)
    mask_level2 = tf.io.parse_tensor(content['mask_level2'], out_type=tf.uint8)
    
    image = tf.reshape(image, shape=(content['height'], content['width'], content['depth']))
    mask_level1 = tf.reshape(mask_level1, shape=(content['height'],content['width'], 1))
    mask_level2 = tf.reshape(mask_level2, shape=(content['height'],content['width'], 1))
    

    return (image, mask_level1, mask_level2)


def parse_tfrecord_nir_swir_image_and_mask(mask_level):

    def parse_tfrecord(example):
        image, mask_level1, mask_level2 = read_tfrecord_raw_example(example)
        # Select only the NIR, SWIR 1 and SWIR 2
        # The model was trained with the order SWIR 2, SWIR 1 and NIR
        image = image[...,-3:]
        image = image[...,::-1]
        
        if int(mask_level) == 1:
            return tf.cast(image, tf.int64), tf.cast(mask_level1, tf.float64)
        
        if int(mask_level) == 2:
            return tf.cast(image, tf.int64), tf.cast(mask_level2, tf.float64)
    
        raise Exception('Unknow mask level')
        # image = image

    return parse_tfrecord

def read_example(example):
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'raw_mask': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(example, data)
  
    height = content['height']
    width = content['width']
    depth = content['depth']
    raw_image = content['raw_image']
    raw_mask = content['raw_mask']
    
    #get our 'feature'-- our image -- and reshape it appropriately
    image = tf.io.parse_tensor(raw_image, out_type=tf.float32)
    image = tf.reshape(image, shape=(height,width,depth))

    # image = image * 10000 / 16280.0
    
    mask = tf.io.parse_tensor(raw_mask, out_type=tf.float32)
    mask = tf.reshape(mask, shape=(height,width))

    return (image, mask)



def open_image_and_mask(paths):
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