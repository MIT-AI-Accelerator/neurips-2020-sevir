# NeurIPS 2020 SEVIR
Code for NeurIPS 2020 SEVIR paper


## Requirements

To test pretrained models, this requires

*`tensorflow >2.1.0`

*`pandas`

*`matplotlib`

The following module is used for extracting training/testing dataset from SEVIR.  This contains data readers and generators:

*(removed for anonymity)

To visualize results with statelines as is done in the paper, a geospatial plotting library is required.  We recommend either of the following:

*`basemap`

*`cartopy`


## Downloading SEVIR

The SEVIR data will be avaialbe online.   Link removed from this README to maintain anonymity.

For the remainder, we assume SEVIR is downloaded or linked under the `data/` directory in the repo.

## Extrating training/testing datasets

The models implemented in the paper are implemented on training data collected prior to June 1, 2019, and testing data collected after June 1, 2019.  These datasets can be extrated from SEVIR by running the following scripts (one for nowcasting, and one for synrad).  Depending on your CPU and speed of your filesystem, these scripts may take several hours to run. 


```
cd src/data

# Generates nowcast training & testing datasets
python make_nowcast_dataset.py --sevir_data ../../data/sevir --sevir_catalog ../../data/CATALOG.csv --output_location ../../data/interim/

# Generate synrad training & testing datasets
python make_synrad_dataset.py --sevir_data ../../data/sevir --sevir_catalog ../../data/CATALOG.csv --output_location ../../data/interim/
```











