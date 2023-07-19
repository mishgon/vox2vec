# vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images

This repository is the official implementation of vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images.

## Requirements

Make sure you have installed [torch](https://pytorch.org/) compatible with your CUDA version. To install other requirements, run

```setup
git clone https://github.com/mishgon/vox2vec.git && cd vox2vec && pip install -e .
```

## The pre-trained model

You can download the pre-trained vox2vec model from [here](https://drive.google.com/file/d/1A27Wucnb4lN22RV8487-qaxCxynKzGkG/view?usp=sharing).

## Evaluation of the pre-trained model

Below, we describe how to evaluate the pre-trained vox2vec model on [the BTCV dataset](https://www.synapse.org/#!Synapse:syn3193805/tables/) [1].

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

Also, prepare empty folders `/path/to/cache` for caching the preprocessed data, and `/path/to/logs` for logging.

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

| Model                      | Avg Dice Score |
| -------------------------- | -------------- |
| FPN3d from scratch         | 0.76           |
| vox2vec linear probing     | 0.64           |
| vox2vec non-linear probing | 0.72           |
| vox2vec fine-tuning        | 0.77           |

## Pre-training

To reproduce the pre-training of vox2vec we need a large dataset, consisting of six publicly available CT datasets: AMOS [2], FLARE2022 [3], NLST [4], NSCLC [5], MIDRC [6]. We use [amid](https://github.com/neuro-ml/amid) package, which provides us with unified interfaces of these datasets as well as many other publicly available medical image datasets. We refer to [amid docs](https://neuro-ml.github.io/amid/0.12.0/) for data downloading instructions.

Since you have prepared the data, run
```warmup
python warmup_cache.py --cache_dir /path/to/cache
```
to debug and warmup the pre-training dataset.

To pretrain the vox2vec model, run
```pretrain
python pretrain.py --btcv_dir /path/to/btcv --cache_dir /path/to/cache/ --log_dir /path/to/logs
```

## Citation
If you found this code helpful, please consider citing:
```
@article{goncharov2023vox2vec,
  title={vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images},
  author={Goncharov, Mikhail and Soboleva, Vera and Kurmukov, Anvar and Pisov, Maksim and Belyaev, Mikhail},
  journal={TODO},
  year={2023}
}
```

## References
[1] Landman, B., et al.: Miccai multiatlas labeling beyond the cranial vault–workshop and challenge. In: Proc. MICCAI Multi-Atlas Labeling Beyond Cranial Vault—Workshop Challenge. vol. 5, p. 12 (2015)

[2] Ji, Y., et al.: Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation. arXiv preprint arXiv:2206.08023 (2022)

[3] Ma, J., et al.: Fast and low-gpu-memory abdomen ct organ segmentation: the
flare challenge. Medical Image Analysis 82, 102616 (2022)

[4] Data from the national lung screening trial (nlst) (2013). https://doi.org/10.7937/TCIA.HMQ8-J677, https://wiki.cancerimagingarchive.net/x/-oJY

[5] Aerts, H., Velazquez, E.R., Leijenaar, R., Parmar, C., Grossmann, P., Cavalho, S., Bussink, J., Monshouwer, R., Haibe-Kains, B., Rietveld, D., et al.: Data from nsclc-radiomics. The cancer imaging archive (2015)

[6] Armato III, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Zhao, B., Aberle, D.R., Henschke, C.I., Hoffman, E.A., et al.: The lung image database consortium (lidc) and image database resource initiative (idri): a completed reference database of lung nodules on ct scans. Medical physics 38(2), 915–931 (2011)
[7] Tsai, E., et al.: Medical imaging data resource center - rsna international covid radiology database release 1a - chest ct covid+ (midrc-ricord-1a) (2020). https://doi.org/10.7937/VTW4-X588, https://wiki.cancerimagingarchive.net/x/DoDTB
