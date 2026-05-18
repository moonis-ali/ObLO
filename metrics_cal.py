import laspy
import shapefile
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import kurtosis, skew
from scipy.interpolate import griddata
from rasterio import features
from skimage.feature import graycomatrix, graycoprops
import os


# Step 1: Load the data
plot_id = 'plot_06'  #Folder name where all the required layers are present
plot_id1 = '6943399'  #File name of ALS point cloud

las_file = '/path/pc/' + plot_id1 + '.las'
shapefile_path = '/path/clipped_shp_files/' + plot_id + '.shp'
dem_file = '/path/plot_layers/' + plot_id + '/dem.tif'
orthophoto_path = '/path/orthophoto/' + plot_id + '.tif'
aspect_file = '/path/plot_layers/' + plot_id + '/aspect.tif'
chm_file = '/path/plot_layers/' + plot_id + '/chm.tif'
slope_file = '/path/plot_layers/' + plot_id + '/slope.tif'
twi_file = '/path/plot_layers/' + plot_id + '/twi.tif'
output_path = '/path/output/' + plot_id +'_tree_metrics.csv'



paths = [
    las_file,
    shapefile_path,
    dem_file,
    orthophoto_path,
    aspect_file,
    chm_file,
    slope_file,
    twi_file,
]

for path in paths:
    if os.path.isfile(path):
        print(f"File exists: {path}")
    else:
        print(f"File does NOT exist: {path}")


# Helper function to calculate GLCM metrics
def calculate_glcm_metrics(band_data, mask, levels=256):
    masked_data = band_data[mask]

    if len(masked_data) == 0:
        return { 'contrast': np.nan, 'correlation': np.nan, 'energy': np.nan, 'homogeneity': np.nan, 'dissimilarity': np.nan, 'mean': np.nan, 'std': np.nan }

    min_val, max_val = masked_data.min(), masked_data.max()
    if max_val > min_val:
        normalized_data = ((masked_data - min_val) / (max_val - min_val) * (levels - 1)).astype(np.uint8)
    else:
        return { 'contrast': np.nan, 'correlation': np.nan, 'energy': np.nan, 'homogeneity': np.nan, 'dissimilarity': np.nan, 'mean': np.nan, 'std': np.nan }

    size = int(np.sqrt(len(normalized_data)))
    normalized_data = normalized_data[:size**2].reshape((size, size))

    glcm = graycomatrix(normalized_data, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    mean = np.mean(normalized_data)
    std = np.std(normalized_data)

    return { 'contrast': contrast, 'correlation': correlation, 'energy': energy, 'homogeneity': homogeneity, 'dissimilarity': dissimilarity, 'mean': mean, 'std': std }



# Helper function to calculate GLDV features
def calculate_gldv_features(band_data, mask, levels=256):
    masked_data = band_data[mask]

    if len(masked_data) == 0:
        return np.nan, np.nan, np.nan, np.nan

    min_val, max_val = masked_data.min(), masked_data.max()
    if max_val > min_val:
        normalized_data = ((masked_data - min_val) / (max_val - min_val) * (levels - 1)).astype(np.uint8)
    else:
        return np.nan, np.nan, np.nan, np.nan

    size = int(np.sqrt(len(normalized_data)))
    normalized_data = normalized_data[:size**2].reshape((size, size))

    # GLDV Angular Second Moment (ASM)
    gldv_asm = np.sum(np.square(normalized_data))

    # GLDV Entropy
    gldv_entropy = -np.sum(normalized_data * np.log(normalized_data + 1e-10))

    # GLDV Mean (equivalent to GLCM Dissimilarity)
    gldv_mean = np.sum(np.abs(normalized_data - np.mean(normalized_data)))

    # GLDV Contrast (equivalent to GLCM Contrast)
    gldv_contrast = np.sum(np.square(normalized_data - np.mean(normalized_data)))

    return gldv_asm, gldv_entropy, gldv_mean, gldv_contrast





# Step 2: Load all required files
# Load point cloud
try:
    las = laspy.read(las_file)
    x, y, z = las.x, las.y, las.z
    intensity = las.intensity
    classification = las.classification
except Exception as e:
    raise ValueError("Error reading the LAS file.") from e
 
 
# Load the DEM
try:
    with rasterio.open(dem_file) as dem_src:
        dem_data = dem_src.read(1)
        dem_transform = dem_src.transform
except Exception as e:
    raise ValueError("Error reading the DEM file.") from e
    
    
# Load the Orthophoto
try:
    with rasterio.open(orthophoto_path) as ortho_src:
        r_band = ortho_src.read(1)
        g_band = ortho_src.read(2)
        b_band = ortho_src.read(3)
        ortho_transform = ortho_src.transform
except Exception as e:
    raise ValueError("Error reading the orthophoto file.") from e
   

# Load the Aspect
try:
    with rasterio.open(aspect_file) as aspect_src:
        aspect_data = aspect_src.read(1)
        aspect_transform = aspect_src.transform
except Exception as e:
    raise ValueError("Error reading the Aspect file.") from e
    
    
# Load the CHM
try:
    with rasterio.open(chm_file) as chm_src:
        chm_data = chm_src.read(1)
        chm_transform = chm_src.transform
except Exception as e:
    raise ValueError("Error reading the CHM file.") from e
    
    
# Load the Slope
try:
    with rasterio.open(slope_file) as slope_src:
        slope_data = slope_src.read(1)
        slope_transform = slope_src.transform
except Exception as e:
    raise ValueError("Error reading the Slope file.") from e    
 
 
# Load the TWI
try:
    with rasterio.open(twi_file) as twi_src:
        twi_data = twi_src.read(1)
        twi_transform = twi_src.transform
except Exception as e:
    raise ValueError("Error reading the Slope file.") from e    
             

# Step 3: Align the DEM with the LiDAR point cloud
x_pixels, y_pixels = ~dem_transform * (x, y)
x_pixels = np.clip(x_pixels, 0, dem_data.shape[1] - 1).astype(int)
y_pixels = np.clip(y_pixels, 0, dem_data.shape[0] - 1).astype(int)
dtm_interpolator = dem_data[y_pixels, x_pixels]
tree_heights = z - dtm_interpolator

# Step 4: Load the shapefile
try:
    sf = shapefile.Reader(shapefile_path)
    tree_polygons = [shape.shape.__geo_interface__ for shape in sf.shapeRecords()]
except Exception as e:
    raise ValueError("Error reading the shapefile.") from e

# Step 5: Initialize metrics storage
metrics = []

# Step 6: Compute metrics for each tree
# Number of density slices and AIH percentiles
num_slices = 10
AIH_percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
elevationPercentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]

# Threshold for Canopy Cover (e.g., points above 2m considered part of the canopy)
canopy_threshold = 2.0

for i, polygon in enumerate(tree_polygons):
    # Extract polygon vertices and other required data
    poly_coords = np.array(polygon['coordinates'][0])
    poly_x, poly_y = poly_coords[:, 0], poly_coords[:, 1]

    if len(poly_coords) < 3:
        print(f"Skipping invalid or empty polygon {i + 1}.")
        continue

    # Compute the centroid
    centroid_x, centroid_y = poly_coords[:, 0].mean(), poly_coords[:, 1].mean()
    
    # Clip ALS points within the polygon
    in_polygon = np.logical_and.reduce([
        x >= poly_x.min(),
        x <= poly_x.max(),
        y >= poly_y.min(),
        y <= poly_y.max()
    ])
    tree_points = np.array(tree_heights[in_polygon])
    tree_points_z = np.array(z[in_polygon])
    tree_intensities = np.array(intensity[in_polygon])

    if len(tree_points) > 0:
        # Metrics Calculation
        max_height = tree_points.max()
        min_height = tree_points.min()
        mean_height = tree_points.mean()
        median_height = np.median(tree_points)
        std_height = tree_points.std()
        var_height = np.var(tree_points)
        cv_height = std_height / mean_height if mean_height != 0 else np.nan

        # AIH and elevation IQR
        AIH75 = np.percentile(tree_points, 75)
        AIH25 = np.percentile(tree_points, 25)
        AIH_IQD = AIH75 - AIH25

        # Median Absolute Deviation
        MADMedian_height = np.median(np.abs(tree_points - median_height))

        # Canopy Relief Ratio (CRR)
        CRR = (mean_height - min_height) / (max_height - min_height)

        # Average Absolute Deviation (AAD)
        AAD_height = np.mean(np.abs(tree_points - mean_height))

        # Density Metrics
        height_range = max_height - min_height
        slice_interval = height_range / num_slices
        density_metrics = []
        for j in range(num_slices):
            lower_bound = min_height + (j * slice_interval)
            upper_bound = lower_bound + slice_interval
            points_in_slice = tree_points[(tree_points >= lower_bound) & (tree_points < upper_bound)]
            density_metrics.append(len(points_in_slice) / len(tree_points))
            
        # Generalized Means
        generalizedMean2 = np.sqrt(np.mean(tree_points_z**2))
        generalizedMean3 = np.cbrt(np.mean(tree_points_z**3))

        # Elevation Percentiles
        elevation_percentiles = np.percentile(tree_points_z, elevationPercentiles)
        Ele25 = elevation_percentiles[4]
        Ele75 = elevation_percentiles[11]
        elevation_IQR = Ele75 - Ele25

        # Skewness and Kurtosis
        n = len(tree_points_z)
        std_z = tree_points_z.std()
        mean_z = tree_points_z.mean()
        skewness_z = np.sum(((tree_points_z - mean_z) / std_z)**3) / n if n > 0 else np.nan
        kurtosis_z = np.sum(((tree_points_z - mean_z) / std_z)**4) / n if n > 0 else np.nan


        ################ Intensity Metrics ######################
        mean_intensity = tree_intensities.mean()
        std_intensity = tree_intensities.std()
        var_intensity = np.var(tree_intensities)
        coeff_variation_intensity = std_intensity / mean_intensity if mean_intensity != 0 else np.nan
        skewness_intensity = ((tree_intensities - mean_intensity) / std_intensity).mean()
        kurtosis_intensity = (((tree_intensities - mean_intensity) / std_intensity)**4).mean()
        
        MADMedian_intensity = np.median(np.abs(tree_intensities - np.median(tree_intensities)))
        AAD_intensity = np.mean(np.abs(tree_intensities - mean_intensity))
        max_intensity = tree_intensities.max()
        min_intensity = tree_intensities.min()
        median_intensity = np.median(tree_intensities)
        
        # Calculate AII Metrics (1%, 5%, ..., 99%)
        sorted_intensities = np.sort(tree_intensities)
        cumulative_intensity = np.cumsum(sorted_intensities) / np.sum(sorted_intensities)

        if len(cumulative_intensity) < 2:
            aii_metrics = [np.nan] * 15  # Assign NaN for missing values
        else:
            aii_metrics = [
                np.interp(x, np.linspace(0, 100, len(cumulative_intensity)), cumulative_intensity)
                for x in [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
            ]

        # Intensity Percentiles
        intensity_percentiles = np.percentile(tree_intensities, AIH_percentiles)
        Int25 = intensity_percentiles[4]
        Int75 = intensity_percentiles[11]
        intensity_IQR = Int75 - Int25

        # Canopy Cover (CC)
        canopy_points = tree_points[tree_points > canopy_threshold]
        CC = len(canopy_points) / len(tree_points) if len(tree_points) > 0 else np.nan

        ####################### Orthophoto metrics ######################
        # Generate a mask for the polygon
        mask_rgb = rasterio.features.geometry_mask([polygon], out_shape=r_band.shape, transform=ortho_transform, invert=True)
    
        # Extract RGB values within the mask
        r_values = r_band[mask_rgb]
        g_values = g_band[mask_rgb]
        b_values = b_band[mask_rgb]
         
        if len(r_values) > 0:
                # Compute mean and standard deviation for each band
                mean_r, std_r = np.mean(r_values), np.std(r_values)
                mean_g, std_g = np.mean(g_values), np.std(g_values)
                mean_b, std_b = np.mean(b_values), np.std(b_values)
        else:
                # Handle empty polygons with NaN
                mean_r = mean_g = mean_b = std_r = std_g = std_b = np.nan 
                
                
        ###################### Slope metrics ######################
        # Generate a mask for the polygon
        mask_slope = rasterio.features.geometry_mask([polygon], out_shape=slope_data.shape, transform=slope_transform, invert=True)

        # Extract slope values within the mask
        slope_values = slope_data[mask_slope]

        if len(slope_values) > 0:
                # Compute mean and standard deviation for slope
                mean_slope, std_slope = np.mean(slope_values), np.std(slope_values)
        else:
                # Handle empty polygons with NaN
                mean_slope = std_slope = np.nan
                
        
        ###################### Aspect metrics ######################
        # Generate a mask for the polygon
        mask_aspect = rasterio.features.geometry_mask([polygon], out_shape=aspect_data.shape, transform=aspect_transform, invert=True)

        # Extract aspect values within the mask
        aspect_values = aspect_data[mask_aspect]

        if len(aspect_values) > 0:
                # Compute mean and standard deviation for slope
                mean_aspect, std_aspect = np.mean(aspect_values), np.std(aspect_values)
        else:
                # Handle empty polygons with NaN
                mean_aspect = std_aspect = np.nan
                
                
        ###################### CHM metrics ######################
        # Generate a mask for the polygon
        mask_chm = rasterio.features.geometry_mask([polygon], out_shape=chm_data.shape, transform=chm_transform, invert=True)

        # Extract chm values within the mask
        chm_values = chm_data[mask_chm]

        if len(chm_values) > 0:
                # Compute mean and standard deviation for slope
                mean_chm, std_chm = np.mean(chm_values), np.std(chm_values)
        else:
                # Handle empty polygons with NaN
                mean_chm = std_chm = np.nan     
         
         
         ###################### DEM metrics ######################
        # Generate a mask for the polygon
        mask_dem = rasterio.features.geometry_mask([polygon], out_shape=dem_data.shape, transform=dem_transform, invert=True)

        # Extract dem values within the mask
        dem_values = dem_data[mask_dem]

        if len(dem_values) > 0:
                # Compute mean and standard deviation for slope
                mean_dem, std_dem = np.mean(dem_values), np.std(dem_values)
        else:
                # Handle empty polygons with NaN
                mean_dem = std_dem = np.nan     
         
         
        ###################### TWI metrics ######################
        # Generate a mask for the polygon
        mask_twi = rasterio.features.geometry_mask([polygon], out_shape=twi_data.shape, transform=twi_transform, invert=True)

        # Extract twi values within the mask
        twi_values = twi_data[mask_twi]

        if len(twi_values) > 0:
                # Compute mean and standard deviation for twi
                mean_twi, std_twi = np.mean(twi_values), np.std(twi_values)
        else:
                # Handle empty polygons with NaN
                mean_twi = std_twi = np.nan     
         
         
        ###################### Textural metrics ######################
        glcm_metrics_r = calculate_glcm_metrics(r_band, mask_rgb)
        glcm_metrics_g = calculate_glcm_metrics(g_band, mask_rgb)
        glcm_metrics_b = calculate_glcm_metrics(b_band, mask_rgb)
        glcm_metrics_chm = calculate_glcm_metrics(chm_data, mask_chm)
 
        # Calculate GLDV features for each band (RGB and CHM)
        gldv_asm_r, gldv_entropy_r, gldv_mean_r, gldv_contrast_r = calculate_gldv_features(r_band, mask_rgb)
        gldv_asm_g, gldv_entropy_g, gldv_mean_g, gldv_contrast_g = calculate_gldv_features(g_band, mask_rgb)
        gldv_asm_b, gldv_entropy_b, gldv_mean_b, gldv_contrast_b = calculate_gldv_features(b_band, mask_rgb)
        gldv_asm_chm, gldv_entropy_chm, gldv_mean_chm, gldv_contrast_chm = calculate_gldv_features(chm_data, mask_chm)

        
         
         
        # Store all metrics
        tree_metrics = {
            'TreeID': i + 1,
            'CentroidX': centroid_x,
            'CentroidY': centroid_y,
            'MaxHeight': max_height,
            'MinHeight': min_height,
            'MeanHeight': mean_height,
            'MedianHeight': median_height,
            'StdHeight': std_height,
            'VarHeight': var_height,
            'CVHeight': cv_height,
            'AIH_IQD': AIH_IQD,
            'MADMedianHeight': MADMedian_height,
            'CRR': CRR,
            'AADHeight': AAD_height,
            'GeneralizedMean2': generalizedMean2,
            'GeneralizedMean3': generalizedMean3,
            'Elevation_IQR': elevation_IQR,
            'SkewnessZ': skewness_z,
            'KurtosisZ': kurtosis_z,
            'MeanIntensity': mean_intensity,
            'StdIntensity': std_intensity,
            'VarIntensity': var_intensity,
            'CoeffVariationIntensity': coeff_variation_intensity,
            'MADMedianIntensity': MADMedian_intensity,
            'AADIntensity': AAD_intensity,
            'MaxIntensity': max_intensity,
            'MinIntensity': min_intensity,
            'MedianIntensity': median_intensity,
            'SkewnessIntensity': skewness_intensity,
            'KurtosisIntensity': kurtosis_intensity,
            'Intensity_IQR': intensity_IQR,
            'CC': CC,
            'MeanR': mean_r,
            'MeanG': mean_g,
            'MeanB': mean_b,
            'StdR': std_r,
            'StdG': std_g,
            'StdB': std_b,
            'MeanSlope': mean_slope,
            'MeanAspect': mean_aspect,
            'MeanCHM': mean_chm,
            'MeanDEM': mean_dem,
            'MeanTWI': mean_twi,
            'StdSlope': std_slope,
            'StdAspect': std_aspect,
            'StdCHM': std_chm,
            'StdDEM': std_dem,
            'StdTWI': std_twi,
            'GLCM_Contrast_R': glcm_metrics_r['contrast'],
            'GLCM_Correlation_R': glcm_metrics_r['correlation'],
            'GLCM_Energy_R': glcm_metrics_r['energy'],
            'GLCM_Homogeneity_R': glcm_metrics_r['homogeneity'],
            'GLCM_Dissimilarity_R': glcm_metrics_r['dissimilarity'],
            'GLCM_Mean_R': glcm_metrics_r['mean'],
            'GLCM_Std_R': glcm_metrics_r['std'],
            'GLCM_Contrast_G': glcm_metrics_g['contrast'],
            'GLCM_Correlation_G': glcm_metrics_g['correlation'],
            'GLCM_Energy_G': glcm_metrics_g['energy'],
            'GLCM_Homogeneity_G': glcm_metrics_g['homogeneity'],
            'GLCM_Dissimilarity_G': glcm_metrics_g['dissimilarity'],
            'GLCM_Mean_G': glcm_metrics_g['mean'],
            'GLCM_Std_G': glcm_metrics_g['std'],
            'GLCM_Contrast_B': glcm_metrics_b['contrast'],
            'GLCM_Correlation_B': glcm_metrics_b['correlation'],
            'GLCM_Energy_B': glcm_metrics_b['energy'],
            'GLCM_Homogeneity_B': glcm_metrics_b['homogeneity'],
            'GLCM_Dissimilarity_B': glcm_metrics_b['dissimilarity'],
            'GLCM_Mean_B': glcm_metrics_b['mean'],
            'GLCM_Std_B': glcm_metrics_b['std'],
            'GLCM_Contrast_CHM': glcm_metrics_chm['contrast'],
            'GLCM_Correlation_CHM': glcm_metrics_chm['correlation'],
            'GLCM_Energy_CHM': glcm_metrics_chm['energy'],
            'GLCM_Homogeneity_CHM': glcm_metrics_chm['homogeneity'],
            'GLCM_Dissimilarity_CHM': glcm_metrics_chm['dissimilarity'],
            'GLCM_Mean_CHM': glcm_metrics_chm['mean'],
            'GLCM_Std_CHM': glcm_metrics_chm['std'],
            'GLDV_Angular_2nd_Moment_R': gldv_asm_r,
            'GLDV_Entropy_R': gldv_entropy_r,
            'GLDV_Mean_R': gldv_mean_r,
            'GLDV_Contrast_R': gldv_contrast_r,
            'GLDV_Angular_2nd_Moment_G': gldv_asm_g,
            'GLDV_Entropy_G': gldv_entropy_g,
            'GLDV_Mean_G': gldv_mean_g,
            'GLDV_Contrast_G': gldv_contrast_g,
            'GLDV_Angular_2nd_Moment_B': gldv_asm_b,
            'GLDV_Entropy_B': gldv_entropy_b,
            'GLDV_Mean_B': gldv_mean_b,
            'GLDV_Contrast_B': gldv_contrast_b,
            'GLDV_Angular_2nd_Moment_CHM': gldv_asm_chm,
            'GLDV_Entropy_CHM': gldv_entropy_chm,
            'GLDV_Mean_CHM': gldv_mean_chm,
            'GLDV_Contrast_CHM': gldv_contrast_chm,
        }

        # Add density metrics and percentiles as separate columns
        for idx, density in enumerate(density_metrics):
            tree_metrics[f'Density_{idx + 1}'] = density
        for idx, percentile in enumerate(AIH_percentiles):
            tree_metrics[f'AIH_{percentile}'] = np.percentile(tree_points, percentile)
        for idx, percentile in enumerate(elevationPercentiles):
            tree_metrics[f'Elevation_{percentile}'] = elevation_percentiles[idx]
        for idx, percentile in enumerate(elevationPercentiles):
            tree_metrics[f'Intensity_{percentile}'] = intensity_percentiles[idx]
        for idx, value in enumerate([1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]):
            tree_metrics[f'AII_{value}'] = aii_metrics[idx]

        metrics.append(tree_metrics)

    else:
        # Handle empty polygons
        tree_metrics = {key: np.nan for key in [
            'TreeID', 'CentroidX', 'CentroidY', 'MaxHeight', 'MinHeight', 'MeanHeight', 'MedianHeight',
            'StdHeight', 'VarHeight', 'CVHeight', 'AIH_IQD', 'MADMedianHeight', 'GeneralizedMean2', 'GeneralizedMean3',
            'CRR', 'AADHeight', 'MeanIntensity', 'StdIntensity', 'VarIntensity',
            'Elevation_IQR', 'SkewnessZ', 'KurtosisZ', 'MADMedianIntensity', 'AADIntensity', 'MaxIntensity',
            'MinIntensity', 'MeanIntensity', 'StdIntensity', 'VarIntensity', 'CoeffVariationIntensity',
            'SkewnessIntensity', 'KurtosisIntensity', 'Intensity_IQR', 'MedianIntensity', 'CC',
            'MeanR', 'MeanG', 'MeanB', 'StdR', 'StdG', 'StdB',
            'MeanSlope', 'MeanAspect', 'MeanCHM', 'MeanDEM', 'MeanTWI',
            'StdSlope', 'StdAspect', 'StdCHM', 'StdDEM', 'StdTWI',
            'GLCM_Contrast_R', 'GLCM_Correlation_R', 'GLCM_Energy_R', 'GLCM_Homogeneity_R',
            'GLCM_Dissimilarity_R', 'GLCM_Mean_R', 'GLCM_Std_R',
            'GLCM_Contrast_G', 'GLCM_Correlation_G', 'GLCM_Energy_G', 'GLCM_Homogeneity_G',
            'GLCM_Dissimilarity_G', 'GLCM_Mean_G', 'GLCM_Std_G',
            'GLCM_Contrast_B', 'GLCM_Correlation_B', 'GLCM_Energy_B', 'GLCM_Homogeneity_B',
            'GLCM_Dissimilarity_B', 'GLCM_Mean_B', 'GLCM_Std_B',
            'GLCM_Contrast_CHM', 'GLCM_Correlation_CHM', 'GLCM_Energy_CHM', 'GLCM_Homogeneity_CHM',
            'GLCM_Dissimilarity_CHM', 'GLCM_Mean_CHM', 'GLCM_Std_CHM',
            'GLDV_Angular_2nd_Moment_R', 'GLDV_Entropy_R', 'GLDV_Mean_R', 'GLDV_Contrast_R',
            'GLDV_Angular_2nd_Moment_G', 'GLDV_Entropy_G', 'GLDV_Mean_G', 'GLDV_Contrast_G',
            'GLDV_Angular_2nd_Moment_B', 'GLDV_Entropy_B', 'GLDV_Mean_B', 'GLDV_Contrast_B',
            'GLDV_Angular_2nd_Moment_CHM', 'GLDV_Entropy_CHM', 'GLDV_Mean_CHM', 'GLDV_Contrast_CHM',
            
        ]}
        for idx in range(num_slices):
            tree_metrics[f'Density_{idx + 1}'] = np.nan
        for idx, percentile in enumerate(AIH_percentiles):
            tree_metrics[f'AIH_{percentile}'] = np.nan
        for idx, percentile in enumerate(elevationPercentiles):
            tree_metrics[f'Elevation_{percentile}'] = np.nan
            tree_metrics[f'Intensity_{percentile}'] = np.nan
	    
        metrics.append(tree_metrics)





# Step 7: Save metrics to CSV
results_df = pd.DataFrame(metrics)
results_df.to_csv(output_path, index=False)

