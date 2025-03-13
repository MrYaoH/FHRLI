# FHRLI

The pytorch implementation of "Investigating All Uncertainties in Hypergraph Representation Learning and Inference" 

# Getting Started

## Dependency

To run our code, the following Python libraries which are required to run our code:

```
python 3.9
pytorch 1.13.1
torch-geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv
```

## Data Preparation

In this work, we use the datasets preprocessed from [ED-HNN](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset).

Then put the downloaded directory under the root folder of this repository. The directory structure should look like:
```
ED-HNN/
  <source code files>
  ...
  raw_data
    20newsW100
    coauthorship
    cocitation
    ...
```

## Training

Before training, please create a folder 'log' for storing the exprimental results.

Please use the following commands to reproduce our results:

<details>

<summary>Cora</summary>

```
python train.py --method FHRLI_EDGNN --dname cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 
--MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean
--restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path> 
```

</details>

<details>

<summary>Citeseer</summary>

```
python train.py --method FHRLI_EDGNN --dname citeseer --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0
--MLP3_num_layers 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean 
--restart_alpha 0.0 --wd 0 --epochs 500 --runs 10
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>


<details>

<summary>Pubmed</summary>

```
python train.py --method FHRLI_EDGNN --dname pubmed --All_num_layers 8 --MLP_num_layers 2 --MLP2_num_layers 2
--MLP3_num_layers 2 --Classifier_num_layers 2 --MLP_hidden 512 --Classifier_hidden 256 --normalization None --aggregate mean
--restart_alpha 0.5 --wd 0 --epochs 500 --runs 10
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>


<details>

<summary>Cora-CA</summary>

```
python train.py --method FHRLI_EDGNN --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0
--MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean 
--restart_alpha 0.0 --wd 0 --epochs 500 --runs 10
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>

<details>

<summary>DBLP-CA</summary>

```
python train.py --method FHRLI_EDGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0
--MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean
--restart_alpha 0.0 --wd 0 --epochs 500 --runs 10
--cuda <cuda_id> --data_dir <data_path> --raw_data_dir <raw_data_path>
```

</details>


