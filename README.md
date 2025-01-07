# Unlocking the diagnostic potential of electrocardiograms through information transfer from cardiac magnetic resonance imaging
This is the official implementation of our work [Unlocking the diagnostic potential of electrocardiograms through information transfer from cardiac magnetic resonance imaging](https://www.sciencedirect.com/science/article/pii/S1361841524003785) (2025). 

<p align="center">
  <img src="./figures/overview.png?raw=true">
</p>

## Instructions

### Masked data modelling
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

## Citation
If you use the provided code, please cite the following work:

```
@article{turgut2025unlocking,
  title={Unlocking the diagnostic potential of electrocardiograms through information transfer from cardiac magnetic resonance imaging},
  author={Turgut, {\"O}zg{\"u}n and M{\"u}ller, Philip and Hager, Paul and Shit, Suprosanna and Starck, Sophie and Menten, Martin J and Martens, Eimo and Rueckert, Daniel},
  journal={Medical Image Analysis},
  pages={103451},
  year={2025},
  publisher={Elsevier}
}
```

## Notice
This project includes third-party software components that are subject to their respective licenses. Detailed information including component names, licenses, and copyright holders is provided in the respective files. Please review the [LICENSE](https://github.com/oetu/MMCL-ECG-CMR/blob/main/LICENSE) file before using or distributing this software.
