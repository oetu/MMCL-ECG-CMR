import os 
import sys
import time
import random

from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from torchvision import transforms

import utils.ecg_augmentations as augmentations
from datasets.ContrastiveECGDataset import ContrastiveECGDataset
from models.ECGSimCLR import ECGSimCLR
from models.BYOL import BYOL
from models.BarlowTwins import BarlowTwins

def train_ecg(hparams, wandb_logger: WandbLogger):
  """
  Training code for lightning model for image only SimCLR. 

  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger

  OUT
  version:      The version under which the model was saved 
                so downstream evaluation uses the correct checkpoint
  """

  pl.seed_everything(hparams.seed)

  hparams.input_size = (hparams.input_channels, hparams.input_electrodes, hparams.time_steps)
  hparams.patch_size = (hparams.patch_height, hparams.patch_width)

  transform = transforms.Compose([
      augmentations.FTSurrogate(phase_noise_magnitude=hparams.ft_surr_phase_noise),
      augmentations.Jitter(sigma=hparams.jitter_sigma),
      augmentations.Rescaling(sigma=hparams.rescaling_sigma),
      augmentations.TimeFlip(prob=0.5),
      augmentations.SignFlip(prob=0.5),
      augmentations.SpecAugment(masking_ratio=0.25, n_fft=120)
  ])
  hparams.transform = transform.__repr__()    
  
  train_dataset = ContrastiveECGDataset(
    data_path=hparams.data_train_ecg, labels_path=hparams.labels_train, 
    transform=transform, augmentation_rate=hparams.augmentation_rate, args=hparams)
  val_dataset = ContrastiveECGDataset(
    data_path=hparams.data_val_ecg, labels_path=hparams.labels_val, 
    transform=transform, augmentation_rate=hparams.augmentation_rate, args=hparams)

  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  # Set log dir and create new version_{version} folder that increments the previous by one.
  basepath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs','ecg')
  if hparams.resume_training:
    version = os.path.dirname(hparams.checkpoint).split('_')[-1]
  else:
    version = hparams.version # str(get_next_version(basepath))
  logdir = os.path.join(basepath,f'version_{version}')
  os.makedirs(logdir,exist_ok=True)

  hparams.version = version

  if hparams.loss.lower() == 'byol':
    model = BYOL(**hparams)
  elif hparams.loss.lower() == 'barlowtwins':
    model = BarlowTwins(**hparams)
  else:
    model = ECGSimCLR(hparams)
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor='ecg.val.loss', mode='min', filename='checkpoint_best_loss', dirpath=logdir))
  #callbacks.append(EarlyStopping(monitor='imaging.val.loss', min_delta=0.001, patience=20, verbose=False, mode='min'))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)

  return version