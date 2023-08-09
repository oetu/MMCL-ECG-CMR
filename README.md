# Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI

This is the official implementation of our paper "Unlocking the Diagnostic Potential of ECG through Knowledge Transfer from Cardiac MRI" (2023).

<p align="center">
  <img src="./figures/overview.png?raw=true">
</p>

## Instructions

### Masked data modeling

Install environment using `conda env create --file /environments/mae.yaml`.
For detailed instructions to run the code for unimodal pre-training, see the [PRETRAIN.md](https://github.com/oetu/mae/blob/1d75ce98082b99accdedbccd00deb5d3eeab8cdb/PRETRAIN.md) of the mae subfolder. 

### Multimodal contrastive learning

Install environment using `conda env create --file /environments/mmcl.yaml`.
For detailed instructions to run the code for multimodal pre-training, see the [README.md](https://github.com/oetu/MMCL-ECG-CMR/blob/main/mmcl/README.md) of the mmcl subfolder. 

### Fine-tuning / inference

Install environment using `conda env create --file /environments/mae.yaml`.
For detailed instructions to run the code for fine-tuning and inference, see the [FINETUNE.md](https://github.com/oetu/mae/blob/1d75ce98082b99accdedbccd00deb5d3eeab8cdb/FINETUNE.md) of the mae subfolder.
