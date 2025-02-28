[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14273617.svg)](https://doi.org/10.5281/zenodo.14273617)

# PSFGAN-GaMPEN
This repository contains source code for the `PSFGAN-GaMPEN` framework (discussed in "Automatic Machine Learning Framework to Study Morphological Parameters of AGN Host Galaxies within $z<1.4$ in the Hyper Supreme-Cam Wide Survey").

[Published Paper](https://doi.org/10.3847/1538-4357/adaec0)

[Arxiv Pre-print](https://arxiv.org/abs/2501.15739)
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

As illustrated in the paper, in each of the following five redshift bins, we trained an individual model of `PSFGAN` and an individual model of `GaMPEN`:
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

At last, change the following block appropriately to check negative fluxes. You may simply insert `or (source == "{target dataset name}")` within the `if` clause in case your flux column has the same format as ours (e.g., `'g_total_flux'`). If there is no flux column you may skip this part and leave this block unchanged.
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
Note: please use a `Python 2.7` environment for `PSFGAN` related tasks.

(Individually in each redshift bin & filter) We now apply our trained `PSFGAN` model on real AGNs in the dataset. It will remove the AGN PS and generate images of recovered host galaxies, which can then be used as inputs for `GaMPEN` models. 

Set the following parameters before proceed:

In `config.py`:
- `learning_rate`: (just for creating corresponding folder names) this should be `0.00005`, `0.000015`, `0.00002`, `0.000008` or `0.000008` for the `low`, `mid`, `high`, `extra`, or `extreme` redshift bin, respectively
- `attention_parameter`: `0.05` (just for creating corresponding folder names)
- `model_path`: `'{location of the trained PSFGAN model in the redshift bin of interest}'` (you can download our trained `PSFGAN` model in each of the five redshift bins & filters from this [Google Drive](https://drive.google.com/drive/folders/1cSxARao_UVPG9RlhYYjp-LvRQOWgA3DB?usp=sharing)) this should point to the .ckpt file
- `beta1`: `0.5`
- `L1_lambda`: `100`
- `sum_lambda`: `0`
- `test_epoch`: this should be `20` for the `low`, `mid`, and `high` redshift bins and `40` for the `extra` and `extreme` redshift bins, respectively.
- `img_size`: `185`
- `train_size`: `185`

Other parameters should be kept the same as in the previous step.

In `model.py`:

If you are using our `low` redshift bin PSFGAN model:
```bash
self.image_00 = tf.slice(self.image, [0, 81, 81, 0], [1, 22, 22, conf.img_channel])
self.cond_00 = tf.slice(self.cond, [0, 81, 81, 0], [1, 22, 22, conf.img_channel])
self.g_img_00 = tf.slice(self.gen_img, [0, 81, 81, 0], [1, 22, 22, conf.img_channel])
```

If you are using our `mid` redshift bin PSFGAN model:
```bash
self.image_00 = tf.slice(self.image, [0, 84, 84, 0], [1, 16, 16, conf.img_channel])
self.cond_00 = tf.slice(self.cond, [0, 84, 84, 0], [1, 16, 16, conf.img_channel])
self.g_img_00 = tf.slice(self.gen_img, [0, 84, 84, 0], [1, 16, 16, conf.img_channel])
```

If you are using our `high` redshift bin PSFGAN model:
```bash
self.image_00 = tf.slice(self.image, [0, 85, 85, 0], [1, 14, 14, conf.img_channel])
self.cond_00 = tf.slice(self.cond, [0, 85, 85, 0], [1, 14, 14, conf.img_channel])
self.g_img_00 = tf.slice(self.gen_img, [0, 85, 85, 0], [1, 14, 14, conf.img_channel])
```

If you are using our `extra` or `extreme` redshift bin PSFGAN model:
```bash
self.image_00 = tf.slice(self.image, [0, 86, 86, 0], [1, 12, 12, conf.img_channel])
self.cond_00 = tf.slice(self.cond, [0, 86, 86, 0], [1, 12, 12, conf.img_channel])
self.g_img_00 = tf.slice(self.gen_img, [0, 86, 86, 0], [1, 12, 12, conf.img_channel])
```

Then, ran `python PSFGAN-GaMPEN/PSFGAN/test.py --mode test` to apply the trained `PSFGAN` model on the dataset in the current redshift bin. 

Outputs (i.e., recovered host galaxy images) will be saved at `PSFGAN-GaMPEN/PSFGAN/{target dataset name}/{the corresponding filter}-band/{stretch_type}_{scale_factor}/lintrain_classic_PSFGAN_{attention_parameter}/lr_{learning_rate}/PSFGAN_output/epoch_{test_epoch}/fits_output/`. The catalog file is stored at `PSFGAN-GaMPEN/PSFGAN/{target dataset name}/{the corresponding filter}-band/{stretch_type}_{scale_factor}/npy_input/catalog_test_npy_input.csv` (created by roouhsc_agn.py in the previous section).
### Applying trained GaMPEN models
(Individually in each redshift bin & filter) We now apply our trained `GaMPEN` model on recovered host galaxy images (outputs of our trained `PSFGAN` model --- see the previous section). It's worth mentioning that each input image (of recovered galaxy) is fed to the `GaMPEN` model for many times (inference). Then we aggregate these inference results to make the final summary catalog for each of the three structural parameters we care about (result aggregation).

Notes: 
1) Please use a `Python 3.7` environment for `GaMPEN` related tasks. See [this page](https://gampen.readthedocs.io/en/latest/Getting_Started.html) for details. You can use `make requirements` and `make check` to quickly set up an environment with all prerequisite packages downloaded.
2) It is not mandatory to use a `GPU` for `GaMPEN` during the steps of inference and result aggregation. That being said, if you would like to use a `GPU`, please install appropriate versions of `CUDA` and `cuDNN`. We assume you have access to a `GPU` and intend to use it for the rest of this tutorial.
3) See [this page](https://gampen.readthedocs.io/en/latest/index.html) for a **comprehensive introduction about `GaMPEN`**. 

Once the environment is set up,  please make sure you have the following directory structure under the `GaMPEN/` folder:
```bash
PSFGAN-GaMPEN/
├── PSFGAN
└── GaMPEN
    ├── docs
    ├── ggt
        ├── __pycache__
        ├── data
            ├── gal_real_0_0.25_gmp
            ├── gal_real_0.25_0.5_gmp
            ├── gal_real_0.5_0.9_gmp
            ├── gal_real_0.9_1.1_gmp
            ├── gal_real_1.1_1.4_gmp
            ├── {target dataset name}
                ├── cutouts
                └── info.csv
            └── {other files and folders}
        ├── losses
        ├── metrics
        ├── models
        ├── modules
        ├── tests
        ├── train
        ├── utils
        ├── visualization
        └── __init__.py
    ├── ggt.egg-info
    ├── mlruns
    ├── intro_image.png
    ├── LICENSE
    ├── Makefile
    ├── README.rst
    ├── requirements.txt
    └── setup.py
```

Here, folders such as `gal_real_0_0.25_gmp` contains scaling files. **These five folders are not there by default and users need to manually download them from the `Scaling Files` folder in this [Google Drive](https://drive.google.com/drive/folders/1cSxARao_UVPG9RlhYYjp-LvRQOWgA3DB?usp=sharing) and put them exactly under `PSFGAN-GaMPEN/GaMPEN/ggt/data/`**. They contain information about inverse transformations (to be processed by `GaMPEN` models) and are therefore **indispensable for the inference and result aggregation steps**.

In addition, please copy all images from the `PSFGAN` output (`PSFGAN-GaMPEN/PSFGAN/{target dataset name}/{the corresponding filter}-band/{stretch_type}_{scale_factor}/lintrain_classic_PSFGAN_{attention_parameter}/lr_{learning_rate}/PSFGAN_output/epoch_{test_epoch}/fits_output/`) to the `PSFGAN-GaMPEN/GaMPEN/ggt/data/{target dataset name}/cutouts/` folder. 

Please also copy the catalog file (`PSFGAN-GaMPEN/PSFGAN/{target dataset name}/{the corresponding filter}-band/{stretch_type}_{scale_factor}/npy_input/catalog_test_npy_input.csv`) to `PSFGAN-GaMPEN/GaMPEN/ggt/data/{target dataset name}/` and **rename it as `info.csv`**. **This `info.csv` file must have a column called "file_name" that contains filenames of images in the `cutouts/` folder**.
#### Inference
Now we have set up the environment for `GaMPEN` and prepared images, catalog and scaling files. It's the time to perform inference using `GaMPEN` models. 

To perform inference using a `GaMPEN` model, simply run the following in an appropriate environment:
```bash
python PSFGAN-GaMPEN/GaMPEN/ggt/modules/inference.py --model_type={some model type} --{flag X} ... --{flag Y} --normalize --{flag Z} ...
```

Note there are many flags one needs to specify in order to run the above command. Flags take the format of `--{flag A}={input A}` if there is an input or `--{flag B}` if no input is needed. Specifically:

(For your reference, please also refer to [this page](https://gampen.readthedocs.io/en/latest/Using_GaMPEN.html#inference) for a detailed description of all possible flags) 

- `model_type`: `'vgg16_w_stn_oc_drp'`
- `model_path`: `'{location of the trained GaMPEN model in the redshift bin of interest}'` (you can download our trained `GaMPEN` model in each of the five redshift bins & filters from this [Google Drive](https://drive.google.com/drive/folders/1cSxARao_UVPG9RlhYYjp-LvRQOWgA3DB?usp=sharing)) this should point to the .pt file
- `output_path`: `'{location to store inference results of the GaMPEN model}'` you may simply set it as `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/` to stick with our conventions
- `data_dir`: `'{location of the input dataset}'` this should be `PSFGAN-GaMPEN/GaMPEN/ggt/data/{target dataset name}/`, which contains the `cutouts/` folder and the `info.csv` catalog as mentioned above
- `cutout_size`: this should be `179` or `143` for the `low` or `mid` redshift bin, respectively. For the `high`, `extra` and `extreme` redshift bins, this should be `95`
- `channels`: `3`
- `slug`: `None`
- `split`: `None`
- `normalize`
- `label_scaling`: the inverse transformation to be performed --- **this must be `'std'` for our trained `GaMPEN` models**
- `batch_size` and `n_workers`: see [this page](https://gampen.readthedocs.io/en/latest/Using_GaMPEN.html#inference)
- `parallel`
- `label_cols`: the column names for the three structural parameters --- **this must be (exactly) 'custom_logit_bt,ln_R_e_asec,ln_total_flux_adus' for our trained `GaMPEN` models**
- `repeat_dims`
- `mc_dropout`: this enables the Monte Carlo dropout during inference and should be left on for our trained `GaMPEN` models
- `n_runs`: how many feedforward inference passes you would like to perform (since the Monte Carlo dropout is enabled, we will essentially feed the same input to a slightly different network during each feedforward inference pass) --- we used to set it to `1000` but it's up to your scientific question at hand as well as available computational resources
- `ini_run_num`: `1`
- `dropout_rate`: for our trained `GaMPEN` models, this should be `0.0004` for the `low` redshift bin, `0.0002` for the `mid` and `high` redshift bins, and `0.00015` for the `extra` and `extreme` redshift bins, respectively. 
- `transform`
- `no-errors`
- `cov_errors`
- `no-labels`: this indicates that we are applying the `GaMPEN` model on a previously unlabelled dataset and therefore it should be left on
- `scaling_data_dir`: `'{location of the scaling file for the inference step}'` this should be `PSFGAN-GaMPEN/GaMPEN/ggt/data/{scaling file folder}/`, where `{scaling file folder}` is `gal_real_0_0.25_gmp`, `gal_real_0.25_0.5_gmp`, `gal_real_0.5_0.9_gmp`, `gal_real_0.9_1.1_gmp` or `gal_real_1.1_1.4_gmp` for the `low`, `mid`, `high`, `extra` or `extreme` redshift bin, respectively.
- `scaling_slug`: this should always be `balanced-dev2` for our trained `GaMPEN` models
  
Run the above command with all flags properly set. If you are following our conventions, inference result should be in the `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/` folder.
#### Result Aggregation
The final step is to aggregate inference results from the previous section --- this will generate a summary catalog that contains statistical facts about the three structural parameters we care about.

Similarly, run the following in an appropriate environment:
```bash
python PSFGAN-GaMPEN/GaMPEN/ggt/modules/result_aggregator.py --data_dir={some data_dir} --{flag X} ... --{flag Y} --unscale --{flag Z} ...
```

Likewise, there are many flags one needs to specify in order to run the above command. Flags take the format of `--{flag A}={input A}` if there is an input or `--{flag B}` if no input is needed. Specifically:

(For your reference, please also refer to [this page](https://gampen.readthedocs.io/en/latest/Using_GaMPEN.html#result-aggregator) for a detailed description of all possible flags) 

- `data_dir`: this should be the location where inference results from `GaMPEN` are stored. If you are following our conventions, this location should be `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/` (i.e., the value of `output_path` flag in `inference.py`)
- `num`: the number of inference results to be aggregated. This generally equals to the value of `n_runs` flag in `inference.py`.
- `out_summary_df_path`: where to put the final summary catalog --- for example `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/summary.csv`
- `out_pdfs_path`: where to put the pdf files from the result aggregation --- for example `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/pdfs/`
- `unscale`: this should always be left on when using our trained `GaMPEN` models since we are going to perform the unscaling
- `scaling_df_path`: this is where the scaling file is located for the result aggregation step. This should be `PSFGAN-GaMPEN/GaMPEN/ggt/data/{scaling file folder}/info.csv`, where `{scaling file folder}` is `gal_real_0_0.25_gmp`, `gal_real_0.25_0.5_gmp`, `gal_real_0.5_0.9_gmp`, `gal_real_0.9_1.1_gmp` or `gal_real_1.1_1.4_gmp` for the `low`, `mid`, `high`, `extra` or `extreme` redshift bin, respectively.
- `drop_old`: the unscaled prediction columns will be dropped if left on

Run the above command with all flags properly set. If you are following our conventions, the aggregated summary catalog should be at `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/summary.csv` and the pdf files should be in the `PSFGAN-GaMPEN/GaMPEN/ggt/data/modules/inference_results/{target dataset name}/pdfs/` folder.
















