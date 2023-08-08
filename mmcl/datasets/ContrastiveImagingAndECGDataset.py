from typing import List, Tuple
import random
import csv

import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from torchvision.io import read_image

import utils.ecg_augmentations as augmentations


class ContrastiveImagingAndECGDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first ECG view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_ecg: str, ecg_random_crop: bool,
      labels_path: str, img_size: int, 
      args) -> None:
      
    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate

    if self.delete_segmentation:
      self.data_imaging = [image[0::2, ...] for image in self.data_imaging]

    self.img_size = img_size
    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size, img_size)),
      transforms.Lambda(lambda x : x.float())
    ])

    # ECG
    self.data_ecg = torch.load(data_path_ecg)
    self.data_ecg = [d.unsqueeze(0) for d in self.data_ecg]
    self.data_ecg = [d[:, :args.input_electrodes, :] for d in self.data_ecg]
    self.ecg_random_crop = ecg_random_crop

    # Classifier
    self.labels = torch.load(labels_path)

    self.args = args
  
  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    im = self.data_imaging[index]
    # im = transforms.CenterCrop(size=int(0.75*self.img_size))(im)
    im = torchvision.transforms.functional.crop(im, top=int(0.21*self.img_size), left=int(0.325*self.img_size), height=int(0.375*self.img_size), width=int(0.375*self.img_size))
    if random.random() < self.augmentation_rate:
      im_aug = (self.transform(im))
    else:
      im_aug = (self.default_transform(im))

    im_orig = self.default_transform(im)
    
    return im_aug, im_orig

  def generate_ecg_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects ECG. The first is always the augmented.
    """
    data = self.data_ecg[index]
    
    if self.ecg_random_crop:
      transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False)
    else:
      transform = augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], start_idx=0, resize=False)
    ecg_orig = transform(data)

    ecg_aug = ecg_orig
    if random.random() < self.augmentation_rate:
      augment = transforms.Compose([augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise),
                                    augmentations.Jitter(sigma=self.args.jitter_sigma),
                                    augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                                    # augmentations.CropResizing(fixed_crop_len=self.args.input_size[-1], resize=False),
                                    #augmentations.TimeFlip(prob=0.33),
                                    #augmentations.SignFlip(prob=0.33)
                                    ])
      ecg_aug = augment(ecg_aug)
    
    return ecg_aug, ecg_orig

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    image_aug, image_orig = self.generate_imaging_views(index)
    ecg_aug, ecg_orig = self.generate_ecg_views(index)
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return image_aug, ecg_aug, label, image_orig, ecg_orig, index
    # # for unet encoder
    # return image_aug[0::2, ...], ecg_aug, label, image_orig[0::2, ...], ecg_orig, index

  def __len__(self) -> int:
    return len(self.data_ecg)