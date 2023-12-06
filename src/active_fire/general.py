import numpy as np
import importlib
from scipy import ndimage
import cv2
from image.converter import band_reflectance_to_radiance, correct_toa_reflectance
from image.sentinel import make_saturated_mask

class ActiveFireIndex:

    def __init__(self, method='baseline'):
        """Instantiate a ActiveFireIndex method using the algorithm name.
        The instance can use the transform method to generates the active fire mask. It expects at least a image as 
        a paremeter, the other parameters will be passed as args and kwargs to the methods.
        Each method can have different kwars in the transform method

        Args:
            method (str, optional): Method name. Defaults to 'baseline'.
        """
        self.method = method
        self.algorithm = self.resolve_algorithm()
        # print(self.algorithm)

    def transform(self, img, *args, **kwargs):
        """Process a given image and generate a active fire mask.
        It will use the transformation of the method defined in the "method" construct.
        Arguments will be passed to the methods

        Args:
            img (np.array): image passed to the algorithm methods

        Returns:
            _type_: _description_
        """
        return self.algorithm.transform(img, *args, **kwargs)

    def resolve_algorithm(self):
        """Instantiate the algorithm by name, without the sufix AFI.
        e.g: 'Muphy' will instantiate the MurphyAFI algorithm. 
        """

        module = importlib.import_module('active_fire.general', '.')
        algorithm = getattr(module, '{}AFI'.format(self.method))

        return algorithm()
    


class LiuAFI:
    def transform(self, img, **kwargs):
        """Generates the active fire mask wusing the Liu, Y., Zhi, W., Xu, B., Xu, W. and Wu, W. (2021). method
        The method proposes a Thermal Anomaly Index (TAI) based on SWIR2, SWIR1 and NIR bands.
        This method may suffer from seamlines caused by clouds and their shadows.


        DOI: https://doi.org/10.1016/j.isprsjprs.2021.05.008
        Args:
            img (np.array): Image with NIR (~0.86nm), SWIR1(~1.61nm) and SWIR2(~2.20nm) bands
        """
            
        b12 = img[:,:,2]
        b11 = img[:,:,1]
        b8 = img[:,:,0]

        # thermal anomaly index
        # tai = (b12 - b11) / b8
        tai = generalized_normalized_difference_index((b12 - b11), b8)

        # Step 1 - clip negative values
        tai_p = tai.copy()
        tai_p[ tai<0 ] = 0

        # Step 2 - compute the mean of tai_p in a 15x15 window 
        tai_mean = cv2.blur(tai_p, ksize=(15,15))

        # generate the initial segmentation mask
        segmentation = (tai_p - tai_mean) > 0.45

        # Step 3 - Create a buffer of 15-pixel around every pixel detected in the previous step
        structure  = np.ones((15,15))
        buffer = ndimage.morphology.binary_dilation(segmentation, structure=structure)
        
        # Step 4 - identify pixels with TAI >= 0.45 in the buffer (15-pixel)
        hta_pixels = ((tai >= 0.45) & buffer)

        # refine the detections
        hta_pixels = hta_pixels & ((b12 - b11) > (b11 - b8)) & (b12 > 0.15)

        # Take in consideration only the saturated pixels in a 8-pixel neighborhood of the previous identified pixels
        structure  = np.ones((3,3))
        buffer = ndimage.morphology.binary_dilation(hta_pixels, structure=structure)
        satured = (b12 >= 1) & (b11 >= 1)
        satured = buffer & satured
        
        # Combine the previous detecion with the satured pixels
        hta_pixels = hta_pixels | satured

        false_alarm_control = ~( (b11 <= 0.05) | (b8 <= 0.01) )

        return hta_pixels & false_alarm_control


class KatoNakamuraAFI:
    
    def transform(self, img, metadata, **kwargs):
        """Generates the active fire mask using the Kato, S. and Nakamura, R. (jul 2017) method.
        It converts the SWIR2 TOA reflectance to radiance in order to apply a threshold, therefore the metadata is needed.

        DOI: 10.1109/IGARSS.2017.8127081
        Args:
            img (np.array): Image with NIR (~0.86nm), SWIR1(~1.61nm) and SWIR2(~2.20nm) bands TOA reflectance
            metadata (dict): Image metadata with solar irradiance and zenith information

        Returns:
            _type_: _description_
        """

        b12 = img[:,:,2]
        b11 = img[:,:,1]
        b8 = img[:,:,0]

        mask = generalized_normalized_difference_index(b12, b8) > 5
        mask = (mask) & (b8 < 0.6) 

        l12 = band_reflectance_to_radiance(b12, 12, metadata)
        mask = mask & (l12 > 0.3)

        false_alarm_control = generalized_normalized_difference_index((b12 - b8),  (b11 - b8))
        false_alarm_control = (1.65 < false_alarm_control) & (false_alarm_control < 33)

        return mask & false_alarm_control



class MurphyAFI:
    
    def transform(self, img, metadata={}, apply_solar_zenith_correction=True):
        """ Generetas an active fire mask using the Murphy et. al. method.
        The image must contain the NIR, SWIR1 and SWIR2 bands (in this order).
        The image be the TOA reflectance corrected by solar zenith. If not corrected, one may use the metadatada and the
        apply_solar_zenith_correction to correct the image.
        The method was orinally proposed to be used in Landsat-8 images, but can also be used in Sentinel-2 images.

        DOI: http://dx.doi.org/10.1016/j.rse.2016.02.027
        Args:
            img (np.array): Image with NIR (~0.86nm), SWIR1(~1.61nm) and SWIR2(~2.20nm) bands
            metadata (dict, optional): Metadata with solar zenith information to apply correction. Only used if apply_solar_zenith_correction=True. Defaults to {}.
            apply_solar_zenith_correction (bool, optional): Correct the TOA solar zenith before generate the active fire mask. If True the metadata must be informed. Defaults to False.

        Returns:
            np.array: Mask with active fire
        """
        
        p7 = img[:,:,2]
        p6 = img[:,:,1]
        p5 = img[:,:,0]
        
        if apply_solar_zenith_correction:
            # Correct the TOA reflectance by the solar zenith
            p7 = correct_toa_reflectance(p7, 12, metadata)
            p6 = correct_toa_reflectance(p6, 11, metadata)
            p5 = correct_toa_reflectance(p5, '8A', metadata)

        unamb_fires = ( generalized_normalized_difference_index(p7, p6) >= 1.4) & (generalized_normalized_difference_index(p7,p5) >= 1.4) & (p7 >= 0.15)

        if np.any(unamb_fires):
            neighborhood = cv2.dilate(unamb_fires.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))).astype(unamb_fires.dtype)
        
            saturated = (make_saturated_mask(p7)) | (make_saturated_mask(p6))
            potential_fires = (((generalized_normalized_difference_index(p6, p5) >= 2) & (p6 >= 0.5)) | saturated)
            potential_fires = potential_fires & neighborhood
            final_mask = (unamb_fires | potential_fires)
        else:
            final_mask = unamb_fires
 
        return (final_mask.astype(bool))
        


def generalized_normalized_difference_index(b1, b2):
    """Compute de Generalized Normalized Difference Index, with is B1/B2
    """
    
    return np.divide(b1, b2, out=np.zeros_like(b1, dtype=np.float32), where=b2!=0)

def normalized_difference_index(b1, b2):
    """Compute the Normalised Difference Index. (b1 - b2) / (b1 + b2).
    """
    ndi = b1 - b2
    div = b1 + b2

    return np.divide(ndi, div, out=np.zeros_like(ndi, dtype=np.float32), where=div!=0)