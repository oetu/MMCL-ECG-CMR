from typing import List, Tuple
import random
import csv

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image

import utils.ecg_augmentations as augmentations


class EvalImagingAndECGDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first ECG view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_ecg: str,
      labels_path: str, img_size: int, live_loading: bool, train: bool) -> None:
      
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # ECG
    self.data_ecg = torch.load(data_path_ecg)
    
    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    if self.live_loading:
      im = read_image(im)
      im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform(im)
    else:
      im = self.default_transform(im)

    ecg = torch.tensor(self.data_ecg[index], dtype=torch.float)
    ecg_transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False)
    ecg = ecg_transform(ecg)

    label = torch.tensor(self.labels[index], dtype=torch.long)

    return (im, ecg), label
    
  def __len__(self) -> int:
    return len(self.data_ecg)