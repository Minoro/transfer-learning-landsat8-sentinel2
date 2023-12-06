import numpy as np

STACK_10M_BANDS_MAP = {
    2: 1,
    3: 2,
    4: 3,
    8: 4,
}

STACK_20M_BANDS_MAP = {
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    '8A': 4,
    '8a': 4,
    11: 5,
    12: 6,
}

STACK_60M_BANDS_MAP = {
    1: 1,
    9: 2,
    10: 3,
}

QUANTIFICATION_VALUE = 10000.0
# QUANTIFICATION_VALUE = 65535.0

SATURATION_VALUE = 65535
NO_DATA_VALUE = 0


def make_saturated_mask(img, saturation_value=SATURATION_VALUE, axis=-1):
    """Create a mask where the pixels are saturated.

    Args:
        img (np.array): Image or band with the values
        saturation_value (int, optional): Value considered saturated. Defaults to SATURATION_VALUE.
        axis (int, optional): Axis to check the values, it must be the image channel axis. Only used if exists an axis for channels. Defaults to 0.

    Returns:
        np.array: mask indicating where saturation occurs
    """
    img = np.asarray(img)

    if len(img.shape) == 2:
        return img == saturation_value

    return np.logical_or.reduce((img == saturation_value), axis=axis)


def make_nodata_mask(img, nodata_values=NO_DATA_VALUE, axis=-1):
    """Create a mask where NODATA values occurs. A true value in the mask identify a pixel with no-data.

    Args:
        img (np.array): Image or band with the values
        nodata_values (int, optional): Value considered as no-data. Defaults to NO_DATA_VALUE.
        axis (int, optional): Axis to check the values, it must be the image channel axis. Only used if exists an axis for channels. Defaults to 0.

    Returns:
        np.array: mask indicating where no-data occurs
    """

    img = np.asarray(img)
    if len(img.shape) == 2:
        return img == nodata_values

    return np.logical_or.reduce((img == nodata_values), axis=axis)