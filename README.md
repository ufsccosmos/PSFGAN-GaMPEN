# PSFGAN-GaMPEN
This repository contains source code for the `PSFGAN-GaMPEN` framework (discussed in "Automatic Machine Learning Framework to Study Morphological Parameters of AGN Host Galaxies within $z<1.4$ in the Hyper Supreme-Cam Wide Survey").
## CONTENTS
```bash
PSFGAN-GaMPEN
    ├── CONTENTS
    ├── Dependencies
    ├── Installation
    └── A brief guide to using our trained PSFGAN-GaMPEN models
        ├── Introduction
        ├── Data splitting
        ├── Real AGN image normalization
        ├── Applying trained PSFGAN models
        └── Applying trained GaMPEN models
            ├── Inference
            └── Result Aggregation
        └── Notes on our trained PSFGAN and GaMPEN models
```
## Dependencies
`Linux` or `OSX`

`Python 2.7` for `PSFGAN` (We will update as soon as the parent source code of PSFGAN is updated to Python 3)

`Python 3.7` for `GaMPEN`

Python modules (including but not limited to): `NumPy`, `SciPy`, `Astropy`, `Pandas`, `TensorFlow`, etc.
## Installation
Clone this repository and change the present working directory to `PSFGAN-GaMPEN/`.

Alternatively, you can download our data from this [Google Drive](https://drive.google.com/drive/folders/1cSxARao_UVPG9RlhYYjp-LvRQOWgA3DB?usp=sharing) (This inclues the source code for `PSFGAN-GaMPEN`, our trained models and scaling files (needed for GaMPEN inference and result aggregation)). 
## A brief guide to using our trained PSFGAN-GaMPEN models
In this section, we will present a quick guide so readers can learn how to effectively use our trained `PSFGAN-GaMPEN` models in a minimal amount of time.
### Introduction
This guide will allow readers to apply our trained models of `PSFGAN-GaMPEN` on real AGNs in the HSC Wide Survey for an estimation of host galaxy magnitude/flux, effective radius and bulge-to-total flux ratio.

As illustrated in the paper, in each of the five redshift bins, we trained an individual model of `PSFGAN` and an individual model of `GaMorNet`:
```bash
Low redshift bin (0 < z < 0.25)
    └── PSFGAN for g-band
        └── GaMPEN for g-band
Mid redshift bin (0.25 < z < 0.5)
    └── PSFGAN for r-band
        └── GaMPEN for r-band
High redshift bin (0.5<z<0.9)
    └── PSFGAN for i-band
        └── GaMPEN for i-band
Extra redshift bin (0.9<z<1.1)
    └── PSFGAN for z-band
        └── GaMPEN for z-band
Extreme redshift bin (1.1<z<1.4)
    └── PSFGAN for y-band
        └── GaMPEN for y-band
```
This is to ensure that we are approximately focusing on images from a restframe wavelength of 450 nm, regardless of the actual redshift of the source.

If you want to apply our model(s) on a dataset, please make sure that:
- Sources in the dataset you're using are imaged in **the corresponding filter** (depending on their redshifts) of the HSC Wide Survey. For example, if your sources are within **0.5<z<0.9**, images from the i-band of the HSC Wide Survey should be fed to our models.
- Each image has a size of **185 by 185 pixels** and the real AGN to-be-studied is located at the center of the image.
- All sources are **real AGNs (active galaxies)** and the AGN PS to host galaxy flux ratio in any filter is equal to or less than 4 (subtracting an extremely luminous AGN PS using `PSFGAN` may leave a much larger residual and can thus confuse `GaMPEN` for accurate parameter estimation.
- We suggest to only use our models for an accurate estimation of bulge-to-total flux ratio for source within **z<0.9** and of galaxy magnitude/flux & effective radius within **z<1.4** (see Figure 10 in our paper). You may still apply our models for your parameters of interest on sources beyond this range, but the result may not be optimal.


### Data splitting
### Real AGN image normalization
### Applying trained PSFGAN models
### Applying trained GaMPEN models
#### Inference
#### Result Aggregation
### Notes on our trained PSFGAN and GaMPEN models
