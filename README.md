# RCGTN
This is a PyTorch implement of the paper submitted to KDD2021.

## Requirements
* python3
* see requirments.txt

## Data Preparation
RCGTN is implemented on those several public datasets.

### Datasets with priori graph structure
* Download PEMS03, PEMS04, PEMS07 and PEMS08 from [STSGCN](https://github.com/Davidham3/STSGCN). Uncompress data files using tar -zxvf data.tar.gz and move them into the data folder.

### Datasets without priori graph structure
* Download Traffic, Electricity datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress data files using tar -zxvf data.tar.gz and move them into the data folder.

Run the following commands to generate train/test/val dataset.

```
# PEMS08
python generate_npy_data.py  --filename=data/PEMS08/PEMS08.npz  --output_dir=data/PEMS08

# Electricity
python generate_txt_data.py  --filename=data/electricity.txt  --output_dir=data/electricity
```

## Model Training  

* PMES03  
`python train.py --dataset pems03 --seq_len 12 --mum_nodes 358 range_size 20 --n_layers 3 `

* PMES04  
`python train.py --dataset pems04 --seq_len 12 --mum_nodes 307 range_size 20 --n_layers 3 `

* PMES07  
`python train.py --dataset pems07 --seq_len 12 --mum_nodes 883 range_size 30 --n_layers 2 `

* PMES08  
`python train.py --dataset pems08 --seq_len 12 --mum_nodes 358 range_size 20 --n_layers 4 `


* Electricity  
`python train.py --dataset electricity --seq_len 24 --mum_nodes 321 range_size 20 --n_layers 3 `


* Traffic  
`python train.py --dataset traffic --seq_len 24 --mum_nodes 862 range_size 30 --n_layers 2 `



