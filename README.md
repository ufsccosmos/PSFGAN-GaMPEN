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

As illustrated in the paper, in each of the following five redshift bins, we trained an individual model of `PSFGAN` and an individual model of `GaMorNet`:
```bash
Low redshift bin (0 < z < 0.25)
    └── PSFGAN for g-band
        └── GaMPEN for g-band
Mid redshift bin (0.25 < z < 0.5)
    └── PSFGAN for r-band
        └── GaMPEN for r-band
High redshift bin (0.5 < z < 0.9)
    └── PSFGAN for i-band
        └── GaMPEN for i-band
Extra redshift bin (0.9 < z < 1.1)
    └── PSFGAN for z-band
        └── GaMPEN for z-band
Extreme redshift bin (1.1 < z < 1.4)
    └── PSFGAN for y-band
        └── GaMPEN for y-band
```
This is to ensure that we are approximately focusing on images from a restframe wavelength of 450 nm, regardless of the actual redshift of the source.

If you want to apply our model(s) on a dataset, please make sure that:
- Sources in the dataset you're using are imaged in **the corresponding filter** (depending on their redshifts) of the HSC Wide Survey. For example, if your sources are within **0.5<z<0.9**, images from the i-band of the HSC Wide Survey should be fed to our models.
- Each image has a size of **185 by 185 pixels** and the real AGN to-be-studied is located at the center of the image.
- All sources are **real AGNs (active galaxies)** and the AGN point sources (PS) to host galaxy flux ratio in any filter is equal to or less than 4 (subtracting an extremely luminous AGN PS using `PSFGAN` may leave a much larger residual and can thus confuse `GaMPEN` for accurate parameter estimation).
- We suggest to only use our models for an accurate estimation of bulge-to-total flux ratio for sources within **z<0.9** and of galaxy magnitude/flux & effective radius within **z<1.4** (see Figure 10 in our paper). You may still apply our models for your parameters of interest on sources beyond this range, but the result may not be optimal.

Once you've checked that these conditions are met, please divide your dataset by the redshift of sources. Then, in each of the redshift bin(s), please first apply our trained model of `PSFGAN` to remove the bright AGN PS. The image of the recovered host galaxy can then be sent to our trained model of `GaMPEN`, which will estimate the three structural parameters (mentioned above) of the host galaxy. **Please keep in mind that in each redshift bin, our trained model of `GaMPEN` can only be applied on images processed by our trained model of `PSFGAN` (i.e., images of PSFGAN-recovered host galaxies)**. One may not apply our trained `GaMPEN` models on inactive galaxies (for instance). Also, models of `PSFGAN` and `GaMPEN` from different redshift bins can not be used together.

From now on, we will mostly use the low redshift bin (for sources within **z<0.25**, using **g-band** images from the HSC Wide Survey) as an example. We will occasionally talk about other redshift bins when necessary (e.g., when summarizing hyperparameter choices). 

Before start, please make sure you have the following directory structure under the `PSFGAN/` folder:
```bash
PSFGAN-GaMPEN/
├── PSFGAN 
    ├── config.py
    ├── data.py
    ├── data_split_agn.py
    ├── galfit.py
    ├── model.py
    ├── normalizing.py
    ├── photometry.py
    ├── roouhsc_agn.py
    ├── test.py
    ├── train.py
    ├── utils.py
    └── {target dataset name}
        └──  g-band
            └── raw_data
                ├── images
                └── {catalog in .csv format}
└── GaMPEN
```
For the target dataset, its raw data images should be stored (in .fits format) in an `image` folder. There should also be a separate catalog file (in .csv format) that contains necessary information of each image. Other columns are optional and users can include as many as they want. Though it would be the user's responsibility to properly handle extra columns when generating new .csv files (see below).
### Data splitting
Essentially, the first step we want to do is to put all raw images in a single folder called `fits_test`. This is just formality and no actual data split is done. We will use `data_split_agn.py` to do so.

In `data_split_agn.py`, set the following parameters to the correct values before proceed:
- `core_path`: full path of the `PSFGAN/` folder
- `galaxy_main`: `core_path` + `'{target dataset name}/'` 
- `filter_strings`: `['g']` (if you are using our trained models not from the low redshift bin, change this appropriately --- `['r']`, `['i']`, `['z']` or `['y']` --- see the previous section)
- `desired_shape`: `[185, 185]` (desired shape of output images in pixels --- **it has to be `[185, 185]` when using our trained models**)
- `--test`: set its default value to the number of galaxies your target dataset has
- `--shuffle`: `1` (`1` to shuffle images before splitting, `0` otherwise)
- `--source`: `'{target dataset name}'` (name of the target dataset --- this should be the same of the corresponding folder name)


### Real AGN image normalization
### Applying trained PSFGAN models
### Applying trained GaMPEN models
#### Inference
#### Result Aggregation
### Notes on our trained PSFGAN and GaMPEN models
