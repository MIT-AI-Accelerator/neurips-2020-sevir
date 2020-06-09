# NeurIPS 2020 SEVIR
Code for NeurIPS 2020 SEVIR paper


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


## Extrating training/testing datasets

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

## Training the `nowcast`/`synrad` models


## Analyzing results

The notebooks under `notebooks` contain code for anaylzing the results of training, and for visualizing the results on sample test cases.










