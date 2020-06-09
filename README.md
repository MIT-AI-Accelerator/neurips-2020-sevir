# NeurIPS 2020 SEVIR
Code for SEVIR paper submitted to NeurIPS 2020


## Requirements

To test pretrained models, this requires

* `tensorflow 2.1.0` or higher
* `pandas`
* `matplotlib`

The following module is used for extracting training/testing dataset from SEVIR.  This contains data readers and generators:

* (removed for anonymity)

To visualize results with statelines as is done in the paper, a geospatial plotting library is required.  We recommend either of the following:

* `basemap`
* `cartopy`


## Downloading SEVIR

The SEVIR data is available online.   Link removed from this README to maintain anonymity during review process.


## Extracting training/testing datasets

The models implemented in the paper are implemented on training data collected prior to June 1, 2019, and testing data collected after June 1, 2019.  These datasets can be extrated from SEVIR by running the following scripts (one for nowcasting, and one for synrad).  Depending on your CPU and speed of your filesystem, these scripts may take several hours to run. 


```
cd src/data

# Generates nowcast training & testing datasets
python make_nowcast_dataset.py --sevir_data ../../data/sevir --sevir_catalog ../../data/CATALOG.csv --output_location ../../data/interim/

# Generate synrad training & testing datasets
python make_synrad_dataset.py --sevir_data ../../data/sevir --sevir_catalog ../../data/CATALOG.csv --output_location ../../data/interim/
```

## Testing pretrained models

Pretrained models used in the paper are located under `models/`.  To run test metrics on these datasets, run the `test_*.py` scripts and point to the pretrained model, and the test dataset.  This shows an example

```
# Test a trained synrad model
python test_synrad.py  models/synrad_mse.h5 data/interim/synrad_testing.h5 test_output.csv
```

## Model training

This section describes how to train the `nowcast` and synthetic weather radar (`synrad`) models yourself.   For the paper, these model were trained using distributed learning over 8 GPUs, however the code in this repo is setup to train on a single GPU.  

The training datasets are pretty large, and running on the full dataset requires a significant amount of RAM.  It is advise to first test the model with `--num_train` set to a low number to start, and increase this to the limits of your system.  Training with all the data may require writing your own generator that batches the data so that it fits in memory.  

### Training `nowcast`


### Training `synrad`

To train `synrad`, make sure the `synrad_training.h5` file is created using the previous step above.  Below we set `num_train` to be only 10,000, but this should be increased for better results.  There are three choices of loss functions configured:  

#### MSE Loss:
```
python train_synrad.py   --num_train 10000  --nepochs 100  --loss_fn  mse  --loss_weights 1.0  --logdir logs/mse_`date +yymmddHHMMSS`
```

#### MSE+Content Loss:
```
python train_synrad.py   --num_train 10000  --nepochs 100  --loss_fn  mse+vgg  --loss_weights 1.0 1.0 --logdir logs/mse_vgg_`date +yymmddHHMMSS`
```

#### cGAN + MAE Loss:
```
python train_synrad.py   --num_train 10000  --nepochs 100  --loss_fn  gan+mae  --loss_weights 1.0 --logdir logs/gan_mae_`date +yymmddHHMMSS`
```

Each of these will write several files into the date-stamped directory in `logs/`, including tracking of metrics, and a model saved after each epoch.  

## Analyzing results

The notebooks under `notebooks` contain code for anaylzing the results of training, and for visualizing the results on sample test cases.










