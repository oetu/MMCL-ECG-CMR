import os
from os.path import join

from typing import List, Tuple
from torch import nn

import torch
from torchvision import transforms

import pytorch_lightning as pl

def get_next_version(root_dir):
  _fs = pl.utilities.cloud_io.get_filesystem(root_dir)

  try:
      listdir_info = _fs.listdir(root_dir)
  except OSError:
    print("Missing logger folder: %s", root_dir)
    return 0

  existing_versions = []
  for listing in listdir_info:
      d = listing["name"]
      bn = os.path.basename(d)
      if _fs.isdir(d) and bn.startswith("version_"):
          dir_ver = bn.split("_")[1].replace("/", "")
          existing_versions.append(int(dir_ver))
  if len(existing_versions) == 0:
      return 0

  return max(existing_versions) + 1

def grab_image_augmentations(img_size: int, target: str) -> transforms.Compose:
  """
  Defines augmentations to be used with images during contrastive training and creates Compose.
  """
  if target.lower() == 'dvm':
    transform = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.0)),
      transforms.Lambda(lambda x: x.float())
    ])
  return transform

def grab_soft_eval_image_augmentations(img_size: int) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1)),
    transforms.Lambda(lambda x: x.float())
  ])
  return transform

def grab_hard_eval_image_augmentations(img_size: int, target: str) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  if target.lower() == 'dvm':
    transform = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1)),
      transforms.Lambda(lambda x: x.float())
    ])
  return transform

def grab_arg_from_checkpoint(ckpt_path: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  ckpt = torch.load(ckpt_path)
  return ckpt['hyper_parameters'][arg_name]

def chkpt_contains_arg(ckpt_path: str, arg_name: str):
  """
  Checks if a checkpoint contains a given argument.
  """
  ckpt = torch.load(ckpt_path)
  return arg_name in ckpt['hyper_parameters']

def prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_ecg', 'data_val_ecg', 
    'data_val_eval_ecg', 'labels_val_eval_ecg', 
    'data_train_eval_ecg', 'labels_train_eval_ecg',
    'data_test_eval_ecg', 'labels_test_eval_ecg',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams['{}_short'.format(hp)] = hparams[hp]
      hparams[hp] = join(db, hparams[hp])

  return hparams

def cos_sim_collate(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
  """
  Collate function to use when cosine similarity of embeddings is relevant. Takes the embeddings returned by the dataset and calculates the cosine similarity matrix for them.
  """
  view_1, view_2, labels, embeddings, thresholds = zip(*data)
  view_1 = torch.stack(view_1)
  view_2 = torch.stack(view_2)
  labels = torch.tensor(labels)
  threshold = thresholds[0]

  cos = torch.nn.CosineSimilarity(dim=0)
  cos_sim_matrix = torch.zeros((len(embeddings),len(embeddings)))
  for i in range(len(embeddings)):
      for j in range(i,len(embeddings)):
          val = cos(embeddings[i],embeddings[j]).item()
          cos_sim_matrix[i,j] = val
          cos_sim_matrix[j,i] = val

  if threshold:
    cos_sim_matrix = torch.threshold(cos_sim_matrix,threshold,0)

  return view_1, view_2, labels, cos_sim_matrix

def calc_logits_labels(out0, out1, temperature=0.1):
  out0 = nn.functional.normalize(out0, dim=1)
  out1 = nn.functional.normalize(out1, dim=1)

  logits = torch.matmul(out0, out1.T) / temperature
  labels = torch.arange(len(out0), device=out0.device)

  return logits, labels