# RCGTN
This is a PyTorch implement of the paper submitted to KDD2021.

## Requirements
* python3
* see requirments.txt

## Data Preparation
RCGTN is implemented on those several public traffic datasets.

### Datasets with priori graph structure
* Download PEMS03, PEMS04, PEMS07 and PEMS08 from [STSGCN](https://github.com/Davidham3/STSGCN). Uncompress data file using tar -zxvf data.tar.gz and move them into the data folder.

### Datasets without priori graph structure
* Download Traffic, Electricity datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress data file using tar -zxvf data.tar.gz and move them into the data folder.

Run the following commands to generate train/test/val dataset.

```
#pems08
python generate_npy_data.py 

# electricity
python generate_txt_data.py
```


