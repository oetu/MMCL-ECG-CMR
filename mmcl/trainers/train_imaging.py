import os 
import sys
import time
import random

from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils.utils import grab_image_augmentations, get_next_version, cos_sim_collate
from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from models.SimCLR import SimCLR
from models.BYOL import BYOL
from models.BarlowTwins import BarlowTwins

def train_imaging(hparams, wandb_logger: WandbLogger):
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

  transform = grab_image_augmentations(hparams.img_size, hparams.target)
  hparams.transform = transform.__repr__()    
  
  train_dataset = ContrastiveImageDataset(
    data=hparams.data_train_imaging, labels=hparams.labels_train, 
    transform=transform, delete_segmentation=hparams.delete_segmentation, 
    augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)
  val_dataset = ContrastiveImageDataset(
    data=hparams.data_val_imaging, labels=hparams.labels_val, 
    transform=transform, delete_segmentation=hparams.delete_segmentation, 
    augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading)

  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)

  # Set log dir and create new version_{version} folder that increments the previous by one.
  basepath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs','imaging')
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
    model = SimCLR(hparams)
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor='imaging.val.loss', mode='min', filename='checkpoint_best_loss', dirpath=logdir))
  #callbacks.append(EarlyStopping(monitor='imaging.val.loss', min_delta=0.001, patience=20, verbose=False, mode='min'))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)

  return version