# MASSH
MApping SSH 


## Examples
### 1. BFN-QG with SWOT & Nadir data
![BFN-QG](images/BFN-QG.png)
This example is taken from a [SSH Mapping Data Challenge (2020a)](https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60).
#### Download the data
The data is hosted [here](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/ocean-data-challenges/dc_data1/catalog.html) with the following directory structure

```
. 
|-- dc_obs
|   |-- 2020a_SSH_mapping_NATL60_topex-poseidon_interleaved.nc
|   |-- 2020a_SSH_mapping_NATL60_nadir_swot.nc 
|   |-- 2020a_SSH_mapping_NATL60_karin_swot.nc
|   |-- 2020a_SSH_mapping_NATL60_jason1.nc
|   |-- 2020a_SSH_mapping_NATL60_geosat2.nc
|   |-- 2020a_SSH_mapping_NATL60_envisat.nc

|-- dc_ref
|   |-- NATL60-CJM165_GULFSTREAM_y****m**d**.1h_SSH.nc

```

To start out create a directory data_Example1
```shell
mkdir data_Example1
```

In this directory, download the *observation* dataset (dc_obs, 285M) using : 
```shell
cd data_Example1
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_obs.tar.gz --no-check-certificate
```
and the *reference* dataset (dc_ref, 11G) using (*this step may take several minutes*) : 

```shell
wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz 
```
Then uncompress the files using `tar -xvf <file>.tar.gz`. You may also use `ftp`, `rsync` or `curl`to donwload the data.  
