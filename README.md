# vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images

This repository is the official implementation of vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images.

## Requirements

Make sure you have installed [torch](https://pytorch.org/) compatible with your CUDA version. To install other requirements run

```setup
git clone https://github.com/mishgon/vox2vec.git && cd vox2vec && pip install -e .
```

## Pre-trained model

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

## Evaluation of the pre-trained model

Below, we describe how to evaluate the pre-trained vox2vec model on [the BTCV dataset](https://www.synapse.org/#!Synapse:syn3193805/tables/).

First, follow the [instructions](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) to download the BTCV dataset. Make sure that the data are unzipped as follows:
```data
/path/to/btcv
    - RawData.zip
    - RawData
        - Training
            - img
                - img0001.nii.gz
                ...
            - label
                - label0001.nii.gz
                ...
        - Testing
            - img
                - img0061.nii.gz
                ...
```

Also, prepare an empty folder `/path/to/cache` for caching the preprocessed data, and `/path/to/logs` for logging.

To evaluate the pre-trained model in the linear and non-linear probing setups, run
```eval
python eval.py --btcv_dir /path/to/btcv --cache_dir /path/to/cache --ckpt /path/to/vox2vec.pt --setup probing --log_dir /path/to/logs/
```

To evaluate the pre-trained model in the fine-tuning setup, run
```eval
python eval.py --btcv_dir /path/to/btcv --cache_dir /path/to/cache --ckpt /path/to/vox2vec.pt --setup fine-tuning --log_dir /path/to/logs/
```

As a baseline, train the same architecture from scratch by running
```eval
python eval.py --btcv_dir /path/to/btcv --cache_dir /path/to/cache --setup from_scratch --log_dir /path/to/logs/
```

You likely get the results close to

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

## Pre-training

To pre-train vox2vec, run this command:

```pretrain
python pretrain.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Citation
