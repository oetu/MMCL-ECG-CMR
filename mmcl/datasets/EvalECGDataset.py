from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import utils.ecg_augmentations as augmentations

class EvalECGDataset(Dataset):
  """"
  Dataset for the evaluation of ECG data
  """
  def __init__(self, data_path: str, labels_path: str, augmentation_rate: float, train: bool, args):
    super(EvalECGDataset, self).__init__()
    self.data = torch.load(data_path)
    self.data = [d.unsqueeze(0) for d in self.data]
    self.data = [d[:, :args.input_electrodes, :] for d in self.data]
    self.labels = torch.load(labels_path)
    self.augmentation_rate = augmentation_rate
    self.train = train
    self.args = args

    self.transform_train = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.args.input_size[-1], resize=False),
      augmentations.FTSurrogate(phase_noise_magnitude=args.ft_surr_phase_noise),
      augmentations.Jitter(sigma=args.jitter_sigma),
      augmentations.Rescaling(sigma=args.rescaling_sigma),
      augmentations.TimeFlip(prob=0.5),
      augmentations.SignFlip(prob=0.5),
      augmentations.SpecAugment(masking_ratio=0.25, n_fft=120)
    ])

    self.transform_val = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.args.input_size[-1], start_idx=0, resize=False)
    ])

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data, label = self.data[index], self.labels[index]

    if self.train and (random.random() <= self.eval_train_augment_rate):
      data = self.transform_train(data)
    else:
      data = self.transform_val(data)
    
    return data, label