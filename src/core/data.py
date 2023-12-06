import tensorflow as tf
import rasterio

g = tf.random.Generator.from_seed(42)


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