import rasterio
# import gdal
import os
import glob
from osgeo import ogr
import json
import math
import numpy as np
import rasterio
import copy 
from joblib import Parallel, delayed


def convert_dir_jp2_to_tiff(input_dir, output_dir = None, verbose=True, n_jobs=1):
    """Convert the JP2000 files in the 'input_dir' directory to TIFF.
    All files with '.jp2' extension will be converted.

    Args:
        input_dir (str): Path with the JP2000 files 
        output_dir (str, optional): Paths where the TIFF files will be stored. If None will save in the 'input_dir' directory. Defaults to None.
        verbose (bool, optional): If True print text showing the progress of the conversion. Defaults to True.
        n_jobs (int, optional): Number of jobs to execute the conversion. Defaults to 1.
    """
    files = glob.glob(os.path.join(input_dir, '*.jp2'))

    num_files = len(files)
    if n_jobs == 1:

        for index, file in enumerate(files, start=1):
            if verbose:
                print('{}/{} - {}'.format(index, num_files, file))
            jp2_to_tiff(file, output_dir)

    else:
        Parallel(n_jobs=n_jobs, verbose=0)(delayed(jp2_to_tiff)(file, output_dir) for file in files)
        

def jp2_to_tiff(file, output_path = None):
    """Convert a JP2000 file to TIFF.
    The output file will be save with the same name as the input file.

    Args:
        file (str): File, with full path, to be converted 
        output_path (str, optional): Path to save the converted file, if None will be save in the same directory as 'file'. Defaults to None.
    """
    dst_dataset = file.replace('.jp2', '.tif')
    
    if output_path is not None:
        file_name = os.path.basename(dst_dataset)
        dst_dataset = os.path.join(output_path, file_name)

    if not os.path.exists(dst_dataset):
        try:
            with rasterio.open(file) as src:
                # Save the image in the tif file
                meta = src.meta
                meta.update(driver='GTiff')
                with rasterio.open(dst_dataset, 'w+', **meta) as dst:
                    dst.write(src.read())
                
            
        except Exception as e:
            print(e)

def build_stacks(rootPath, output_path):
    """Read the Sentinel bands and write TIFF images with the same spatial resolution.
    Each band must be in a separeted file in a JP2000 format.
    The bands will be assembled in a single file (named stack) with the others with the same spatial resolution.
    The stack with 10m will have the bands 2, 3, 4 and 8.
    The stack with 20m will have the bands 5, 6, 7, 8A, 11 and 12.
    The stack with 60m will have the bands 1, 9 and 10.
    The output stacks will have the sufix '_Xm_stack.tif' where X is the spacial resolution (10, 20 or 60).
    

    Args:
        rootPath (str): Path with Sentinel image bands in format JP2000.
        output_path (str): Path to write the image stacks
    """
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # find the diffetent Sentinel images based on the first band
    pattern = '*_B01.jp2'

    files = glob.glob(os.path.join(rootPath, pattern))
    files.sort()

    for ind,file in enumerate(files):


        print(str(ind+1) + '/' + str(len(files)), '-', file[:-8])
        
        file_list10 = []
        file_list20 = []
        file_list60 = []
        
        file_out = file[:-8] + '_10m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)

        if not os.path.exists(file_out):
            # convert the bands to TIFF
            jp2_to_tiff(file[:-8] + '_B02.jp2')
            jp2_to_tiff(file[:-8] + '_B03.jp2')
            jp2_to_tiff(file[:-8] + '_B04.jp2')
            jp2_to_tiff(file[:-8] + '_B08.jp2')

            # Select the bands with 10m of spacial resolution
            file_list10.append(file[:-8] + '_B02.tif')
            file_list10.append(file[:-8] + '_B03.tif')
            file_list10.append(file[:-8] + '_B04.tif')
            file_list10.append(file[:-8] + '_B08.tif')
            
            # Read the metadata from one band
            with rasterio.open(file_list10[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list10))
                meta.update(nodata = 0)                

            # Write the stack with 10m resolution using the metadata
            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list10, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))   
                                
            
        file_out = file[:-8] + '_20m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)
        if not os.path.exists(file_out):
            # convert the bands to TIFF
            jp2_to_tiff(file[:-8] + '_B05.jp2')
            jp2_to_tiff(file[:-8] + '_B06.jp2')
            jp2_to_tiff(file[:-8] + '_B07.jp2')
            jp2_to_tiff(file[:-8] + '_B8A.jp2')
            jp2_to_tiff(file[:-8] + '_B11.jp2')
            jp2_to_tiff(file[:-8] + '_B12.jp2')
            
            # Select the bands with 20m of spacial resolution
            file_list20.append(file[:-8] + '_B05.tif')
            file_list20.append(file[:-8] + '_B06.tif')
            file_list20.append(file[:-8] + '_B07.tif')
            file_list20.append(file[:-8] + '_B8A.tif')
            file_list20.append(file[:-8] + '_B11.tif')
            file_list20.append(file[:-8] + '_B12.tif')
            
            # Read the metadata from one band
            with rasterio.open(file_list20[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list20))          
                meta.update(nodata = 0)                

            # Write the stack with 20m resolution using the metadata
            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list20, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))   
                        
        
        file_out = file[:-8] + '_60m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)
        if not os.path.exists(file_out):
            # convert the bands to TIFF
            jp2_to_tiff(file[:-8] + '_B01.jp2')
            jp2_to_tiff(file[:-8] + '_B09.jp2')
            jp2_to_tiff(file[:-8] + '_B10.jp2')

            # Select the bands with 60m of spacial resolution            
            file_list60.append(file[:-8] + '_B01.tif')
            file_list60.append(file[:-8] + '_B09.tif')
            file_list60.append(file[:-8] + '_B10.tif')
            
            # Read the metadata from one band
            with rasterio.open(file_list60[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list60))                
                meta.update(nodata = 0)                

            # Write the stack with 20m resolution using the metadata
            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list60, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))  



def get_gml_geometry(gml_file):
    """Read a GML file and get the geometry of the file.

    Args:
        gml_file (str): Path to GML file

    Returns:
        list: List with the geometries of the GML file
    """
    reader = ogr.Open(gml_file)
    layer = reader.GetLayer()
    if layer is None:
        return None

    geometries = []
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        if feature is not None:
            json_feature = json.loads(feature.ExportToJson())
            geometries.append(json_feature['geometry'])
    
    return geometries


def gml_to_mask(gml_file, mask_shape, transform):
    """Transform the geometry in the GML file in a binary mask.

    Args:
        gml_file (str): Path to GML file
        mask_shape (tuple): Shape of the output mask, it must have two elements (height, width).
        transform (rasterio.transform): Rasterio Transform that represents the geolocation of the mask

    Returns:
        np.array: A mask as numpy array with the shape of mask_shape.
    """
    geometry = get_gml_geometry(gml_file)
    mask = np.ones(mask_shape, dtype=np.bool)
    if geometry is not None:
        mask = rasterio.features.geometry_mask(geometry, mask_shape, transform)

    return mask

def get_cloud_mask(gml_cloud_mask_path, mask_shape, transform):
    cloud_geometry = get_gml_geometry(gml_cloud_mask_path)
    cloud_mask = np.ones(mask_shape, dtype=np.bool)
    if cloud_geometry is not None:
        cloud_mask = rasterio.features.geometry_mask(cloud_geometry, mask_shape, transform)

    return cloud_mask

def reflectance_to_radiance(img_stack, metadata):

    buffered_image = copy.deepcopy(img_stack)
    for band in img_stack.buffer:

        band_value = img_stack.read(band)
        radiance = band_reflectance_to_radiance(band_value, band, metadata)
        
        buffered_image.set_band(band, radiance)
    
    return buffered_image

def band_reflectance_to_radiance(band_value, band, metadata):
    """
    Convert the reflectance value to radiance value using the information in the metadata
    https://gis.stackexchange.com/questions/285996/convert-sentinel-2-1c-product-from-reflectance-to-radiance
    """

    # get the band name in the metadata
    band_id = str(band)
    metadata_key = 'B' + str(band)
    if metadata_key in metadata:
        band_id = str(metadata[metadata_key])

    solar_irradiance = float(metadata['solar_irradiance_' + band_id])
    
    solar_angle_correction = math.radians(float(metadata['zenith_' + band_id]))
    solar_angle_correction = math.cos(solar_angle_correction)

    # d2 = 1.0 / float(metadata['U'])
    # radiance = (band_value * solar_irradiance * solar_angle_correction ) / (math.pi * d2)

    radiance = (band_value * solar_irradiance * solar_angle_correction * float(metadata['U'])) / math.pi

    return radiance


def correct_toa_reflectance(toa_value, band, metadata):
    """Correct the TOA value by the solar zenith angle

    Args:
        toa_value (np.ndarray): TOA reflectance values without correction
        band (str): Name of the band being converted, used to recover the solar zenith angle from the metadata 
        metadata (dict): Image metadata, with 'zenith_<band_id>' value 
    """

    # get the band name in the metadata
    band_id = str(band)
    metadata_key = 'B' + str(band)
    if metadata_key in metadata:
        band_id = str(metadata[metadata_key])


    cos_sza = math.cos(math.radians(float(metadata['zenith_'+band_id])))
    return toa_value / cos_sza