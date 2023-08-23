# Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI

This is the official implementation of our paper [Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI](https://arxiv.org/abs/2308.05764) (2023). 

<p align="center">
  <img src="./figures/overview.png?raw=true">
</p>

If you find the code useful, please cite

```
@article{turgut2023unlocking,
  title={Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI},
  author={Turgut, {\"O}zg{\"u}n and M{\"u}ller, Philip and Hager, Paul and Shit, Suprosanna and Starck, Sophie and Menten, Martin J and Martens, Eimo and Rueckert, Daniel},
  journal={arXiv preprint arXiv:2308.05764},
  year={2023}
}
```

## Instructions

### Masked data modeling

Install environment using `conda env create --file environments/mae.yaml`. 
Install [timm](https://github.com/oetu/pytorch-image-models/tree/3dbe2c484b7c5e44097427d5fcb50338df895b31/timm) library using `pip install -e pytorch-image-models`.
For detailed instructions to run the code for unimodal pre-training, see the [PRETRAIN.md](https://github.com/oetu/mae/blob/1d75ce98082b99accdedbccd00deb5d3eeab8cdb/PRETRAIN.md) of the mae subfolder. 

### Multimodal contrastive learning

Install environment using `conda env create --file environments/mmcl.yaml`.
Install [timm](https://github.com/oetu/pytorch-image-models/tree/3dbe2c484b7c5e44097427d5fcb50338df895b31/timm)  library using `pip install -e pytorch-image-models`.
For detailed instructions to run the code for multimodal pre-training, see the [README.md](https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/README.md) of the mmcl subfolder. 

### Fine-tuning / inference

Install environment using `conda env create --file environments/mae.yaml`.
Install [timm](https://github.com/oetu/pytorch-image-models/tree/3dbe2c484b7c5e44097427d5fcb50338df895b31/timm)  library using `pip install -e pytorch-image-models`.
For detailed instructions to run the code for fine-tuning and inference, see the [FINETUNE.md](https://github.com/oetu/mae/blob/1d75ce98082b99accdedbccd00deb5d3eeab8cdb/FINETUNE.md) of the mae subfolder.
