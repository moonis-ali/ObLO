# ObLO: Object-Based LiDAR-Orthophoto Upscaling Framework

This repository contains the core implementation of the **ObLO (Object-Based LiDAR-Orthophoto)** framework developed for large-scale individual tree volume estimation using multisource remote sensing data.

The repository currently includes:

* `feature_selection_n_upscaling.ipynb` – notebook for feature selection and upscaling
* `feature_extraction.py` – script for ALS, orthophoto, and topographic feature extraction from segmented crown objects

The framework integrates:

* TLS-derived tree volumes as reference/training data
* ALS-derived structural metrics
* Orthophoto-derived spectral and texture features
* Machine learning-based upscaling models

---

# Overview

Conventional upscaling approaches often rely on pixel- or grid-based representations, which may fragment tree crowns or merge multiple trees into a single modelling unit. ObLO addresses this limitation by using object-based crown representations derived from ALS canopy height models (CHMs) and orthophotos.

The current implementation focuses on:

* Object-level feature extraction
* Multi-stage feature selection
* Development and validation of upscaling models

---

# Repository Contents

## 1. `metrics_cal.py`

This script performs object-level feature extraction using:

* ALS point clouds
* CHM
* DEM
* Slope
* Aspect
* TWI
* Orthophotos
* Crown polygon shapefiles

### Extracted Features

The script computes:

### ALS Structural Metrics

* Height statistics
* Height percentiles
* AIH metrics
* Density metrics
* Canopy Relief Ratio
* Generalized Means
* Intensity metrics

### Texture Features

* GLCM metrics
* GLDV metrics

### Raster-Based Features

* CHM statistics
* DEM statistics
* Slope statistics
* Aspect statistics
* TWI statistics

### Orthophoto Features

* RGB mean and standard deviation
* Texture descriptors

The output is exported as a CSV feature table for subsequent modelling.

---

## 2. `Upscaling_RF.ipynb`

This notebook contains the workflow for:

* Feature preprocessing
* Feature selection
* Correlation filtering
* RF importance ranking
* Recursive Feature Elimination with Cross Validation (RFECV)
* Machine learning model training
* Model validation
* Accuracy assessment
* 

### Evaluation Metrics

* R²
* RMSE
* Relative RMSE (rRMSE)

---

# Workflow Summary

The overall ObLO framework consists of:

1. TLS-based reference volume generation
2. Crown object delineation
3. Object-level feature extraction
4. Multi-stage feature selection
5. Machine learning-based upscaling
6. Validation and accuracy assessment

---

# Required Input Data

The workflow requires:

* ALS point cloud (.las)
* Crown polygon shapefiles
* Orthophotos
* CHM raster
* DEM raster
* Slope raster
* Aspect raster
* TWI raster
* TLS-derived reference tree volumes

---

# Python Libraries

* laspy
* rasterio
* numpy
* pandas
* scipy
* scikit-image
* scikit-learn
* xgboost

---


