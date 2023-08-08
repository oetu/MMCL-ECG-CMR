from typing import Any, Tuple

import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils.ecg_augmentations as augmentations


class ContrastiveECGDataset(Dataset):
  """Fast EEGDataset (fetching prepared data and labels from files)"""
  def __init__(self, data_path: str, labels_path: str, transform=None, augmentation_rate: float=1.0, args=None) -> None:
    """
    data_path:            Path to torch file containing images
    labels_path:          Path to torch file containing labels
    transform:            Compiled torchvision augmentations
    """
    self.transform = transform
    self.augmentation_rate = augmentation_rate

    self.default_transform = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.input_size[-1], resize=False)
    ])

    # # "CLOCS" Kiyasseh et al. (2021) (https://arxiv.org/pdf/2005.13249.pdf)
    # self.default_transform_1 = transforms.Compose([
    #   augmentations.CropResizing(fixed_crop_len=2500, start_idx=0, resize=False)
    # ])
    # self.default_transform_2 = transforms.Compose([
    #   augmentations.CropResizing(fixed_crop_len=2500, start_idx=2500, resize=False)
    # ])

    self.data_ecg = torch.load(data_path) # load to ram
    self.data_ecg = [d.unsqueeze(0) for d in self.data_ecg]
    self.data_ecg = [d[:, :args.input_electrodes, :] for d in self.data_ecg]

    self.labels = torch.load(labels_path) # load to ram

  def __len__(self) -> int:
    """
    return the number of samples in the dataset
    """
    
    return len(self.labels)

  def __getitem__(self, idx) -> Tuple[Any, Any]:
    """
    returns two augmented views of one signal and its label
    """
    data = self.data_ecg[idx]
    
    view_1 = self.transform(self.default_transform(data))
    view_2 = self.default_transform(data)
    if random.random() < self.augmentation_rate:
      view_2 = self.transform(view_2)
  
    return view_1, view_2, self.labels[idx], idx