# RITA
This Repo contains code and data for [Jiaming, Liang, et al. "RITA: Group Attention is All You Need for Timeseries Analytics."]() If you use RITA in your project, please cite the following paper:

```
a
```

![overview_0](https://drive.google.com/uc?id=1YkwqSqtJQDBZQqakRySXelDrl8juNPOg)

![overview_1](https://drive.google.com/uc?id=1DJDI-Bxt76ZIgKeV2P9Kyyk_0oSFq48T)



- [RITA](#rita)
  - [Installation](#installation)
  - [Data](#data)
  - [Model Running](#model-running)
    - [Full-label Training](#full-label-training)
    - [Pretrain & Few-label Finetuning](#pretrain--few-label-finetuning)
  




## Installation
First, clone and install the git repository:
```
git clone
cd rita/
```

Then, install the required packages through `pip`.
```
pip install -r requirements.txt
```


## Data
The zip file of preprocessed dataset can be acess [here](). Please download the zip file and put it under the directory `rita/`.
Then perform the following commands to unzip dataset files.
```
tar -zxvf rita_dataset.tar.gz
```

## Model Running
Go to the under directory `code`:
```
cd code/
```

The running command is as following:
```
python run.py --dataset DATASET --mode MODE --label_size LABLE_SIZE [--pretrained_path PRETRAINED_PATH]
```
`DATASET` describes the dataset, which should be among `[WISDM,RWHAR,HHAR,ECG]`. `MODE` describes the running mode, which should be among `[pretrain,train,finetune]`. `LABLE_SIZE` describes the size of labels used in training stage, which should be among `[full,few]`. `PRETRAINED_PATH` is requried when `MODE=finetune`; it indicates the path to pretrained checkpoint to start with.



### Full-label Training

To perform full-label training, set `MODE` to `train` and `LABLE_SIZE` to `full`. For example, the command of full-label training on dataset WISDM is:
```
python run.py --dataset WISDM --mode train --label_size full
```


### Pretrain & Few-label Finetuning

To perform self-superviesed pretraining, set `MODE` to `pretrain`. For example, the command of full-label training on dataset WISDM is:
```
python run.py --dataset WISDM --mode pretrain --label_size full
```

The pretrained checkpoints are stored in directory `rita/code/checkpoints`. After pretraining, the checkpoints can be copied and saved for further finetuning.

To perform few-label finetuning, set `MODE` to `finetune` and set `PRETRAINED_PATH` to the pretrained checkpoint's path, which is either obtained by performing pretraining as described above or downloading the pretrained checkpoints [here](). For example, the command of few-label finetuning on dataset WISDM is:
```
python run.py --dataset WISDM --mode finetune --label_size few --pretrained_path WISDM_PRETRAINED_PATH
```
