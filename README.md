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
- `galaxy_main`: `core_path` + `'{target dataset name}/'` (users can name this folder --- once it is chosen it should be fixed)
- `filter_strings`:  `['g']`, `['r']`, `['i']`, `['z']` or `['y']` when you are using our trained models in the `low`, `mid`, `high`, `extra`, or `extreme` redshift bin, respectively --- see the previous section
- `desired_shape`: `[185, 185]` (desired shape of output images in pixels --- **it has to be `[185, 185]` when using our trained models**)
- `--test`: set its default value to the number of galaxies your target dataset has
- `--shuffle`: `1` (`1` to shuffle images before splitting, `0` otherwise)
- `--source`: `'{target dataset name}'` (name of the target dataset --- this should be the same of the corresponding folder name)

You should also add appropriate codes at the ends of the following two blocks to process catalogs in `{target dataset name}/`:
```bash
        elif source == "liu":
            column_list = ['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                          'z', 
                          filter_string + '_total_flux']
```
```bash
                elif source == "liu":
                    test_catalogs[i] = test_catalogs[i].append({'object_id': obj_id,
                    'ra': current_row['ra'],
                    'dec': current_row['dec'],
                    'specz_redshift': current_row['specz_redshift'],
                    'specz_flag_homogeneous': current_row['specz_flag_homogeneous'],
                    'z': current_row['z'],
                    filter_strings[i] + '_total_flux': (current_row[filter_strings[i] + '_cmodel_flux'])*nJy_to_adu_per_AA_filters[i]}, ignore_index=True)
```
`data_split_agn.py` creates a new catalog from the old (`{catalog in .csv format}` under `raw_data/`). The first block determines names of the columns in the new catalog while the second block determines how each column in the new catalog is related to columns from the old catalog.

At last, change the following block appropriately to process raw images. You may simply insert `or (source == "{target dataset name}")` within the `if` clause to stick with our conventions.
```bash
    for row_num in row_num_list:
        if (source == "nair") or (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75") or (source == "povic_0_0.5") or (source == "povic_0.5_0.75") or (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0") or (source == "liu"):
            obj_id = int(row_num)

        # Read the images
        images = []
        for i in range(num_filters):
            if (source == "nair") or (source == "gabor_0.3_0.5") or (source == "gabor_0.5_0.75") or (source == "povic_0_0.5") or (source == "povic_0.5_0.75") or (source == "stemo_0.2_0.5") or (source == "stemo_0.5_1.0") or (source == "liu"):
                fits_path = '%s/%s-cutout-*.fits' % (hsc_folders[i], obj_id)
            file = glob.glob(fits_path)[0]
            image = fits.getdata(file)
            images.append(image)
```

Once these parameters are properly set, ran `python PSFGAN-GaMPEN/PSFGAN/data_split_agn.py`.
Corresponding folders and their associated catalogs will be created.
### Real AGN image normalization
The next step is to normalize all images of real AGNs in the `fits_test` folder using the chosen stretch function (see below). 

Certain parameters need to be properly set before we proceed:

In `config.py`:
- `redshift`:  `'{target dataset name}'` 
- `filters_`:  `['g']`, `['r']`, `['i']`, `['z']` or `['y']` for the `low`, `mid`, `high`, `extra`, or `extreme` redshift bin, respectively.
- `stretch_type` and `scale_factor`: `'asinh'` and `50` (**please keep these values fixed as so**)

Also, after the following block:
```bash
    if redshift == 'stemo_0.5_1.0':
        pixel_max_value = 1000
```
One should add:
```bash
    if redshift == '{target dataset name}':
        pixel_max_value = {pixel_max_value}
```
Here, `{pixel_max_value}` denotes the largest pixel value allowed (pre-normliazation) --- this should be `1500`, `500`, `2000`, `750` or `750` if you are using our trained `PSFGAN` models in the `low`, `mid`, `high`, `extra`, or `extreme` redshift bin, respectively. **This value is subject to the trained model you want to use and thus can not be adjusted.** 

In `roouhsc_agn.py`:
- `--source`:  `'{target dataset name}'` (name of the target dataset --- this should be the same of the corresponding folder name)
- `--crop`: `0` (set this to be zero so images are not cropped during normalization)

You should also add appropriate codes at the ends of the following two blocks to properly process catalogs:
```bash
    elif source == 'liu':
        for filter_string in filters_string:
            column_list=['object_id', 'ra', 'dec', 'specz_redshift', 'specz_flag_homogeneous',
                         'z',
                         'galaxy_total_flux_' + filter_string]
            catalog_test_npy_input = pandas.DataFrame(columns=column_list)        
            catalog_test_npy_inputs.append(catalog_test_npy_input)
```
```bash
            elif source == 'liu':
                catalog_per_index = catalog_test_npy_inputs[f_index]
                catalog_per_index = catalog_per_index.append({'object_id': image_id,
                                                              'ra': obj_line['ra'].item(),
                                                              'dec': obj_line['dec'].item(),
                                                              'specz_redshift': obj_line['specz_redshift'].item(),
                                                              'specz_flag_homogeneous': obj_line['specz_flag_homogeneous'].item(),
                                                              'z': obj_line['z'].item(),
                                                              'galaxy_total_flux_' + filters_string[f_index]:
                                                              obj_line[filters_string[f_index] + '_total_flux'].item()}
                                                              , ignore_index=True)
                catalog_test_npy_inputs[f_index] = catalog_per_index
```
`roouhsc_agn.py` creates a new catalog from the old (`catalog_test.csv` under `{target dataset name}/{the corresponding filter}-band/`). The first block determines names of the columns in the new catalog while the second block determines how each column in the new catalog is related to columns from the old catalog.

At last, change the following block appropriately to check negative fluxes. You may simply insert `or (source == "{target dataset name}")` within the `if` clause to stick with our conventions. If there is no flux column you may skip this part and leave this block unchanged.
```bash
            if (source == 'nair') or (source == 'gabor_0.3_0.5') or (source == 'gabor_0.5_0.75') or (source == 'povic_0_0.5') or (source == 'povic_0.5_0.75') or (source == 'stemo_0.2_0.5') or (source == 'stemo_0.5_1.0') or (source == 'liu'):
                flux = obj_line[filters_string[f_index] + '_total_flux'].item()
                if flux < 0:
                    print(filters_string[f_index] + '_total_flux' + ' value in catalog is negative!')
                    continue
```

Once all parameters are set, ran the following to normalize all images in the `fits_test` folder using the chosen stretch function:
```bash
python PSFGAN-GaMPEN/PSFGAN/roouhsc_agn.py 
```
Corresponding folders and associated catalogs will be created. 
### Applying trained PSFGAN models
### Applying trained GaMPEN models
#### Inference
#### Result Aggregation

