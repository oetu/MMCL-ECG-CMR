import os 
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, get_next_version
from datasets.ContrastiveImagingAndECGDataset import ContrastiveImagingAndECGDataset
from models.MultimodalSimCLR import MultimodalSimCLR
from models.MultimodalSimCLR_MultipleLR import MultimodalSimCLR_MultipleLR


def train_multimodal(hparams, wandb_logger):
  """
  Train code for multimodal SimCLR contrastive model. 
  
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
  
  train_dataset = ContrastiveImagingAndECGDataset(
    hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
    hparams.data_train_ecg, True,
    hparams.labels_train, hparams.img_size,
    hparams)
  val_dataset = ContrastiveImagingAndECGDataset(
    hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
    hparams.data_val_ecg, False,
    hparams.labels_val, hparams.img_size,
    hparams)

  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)


  # Set log dir and create new version_{version} folder that increments the previous by one.
  basepath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', 'multimodal')
  if hparams.resume_training:
    version = os.path.dirname(hparams.checkpoint).split('_')[-1]
  else:
    version = hparams.version # str(get_next_version(basepath))
  logdir = os.path.join(basepath,f'version_{version}')
  os.makedirs(logdir,exist_ok=True)

  # hparams.version = version
  hparams.input_size = (hparams.input_channels, hparams.input_electrodes, hparams.time_steps)
  hparams.patch_size = (hparams.patch_height, hparams.patch_width)
  if hparams.multiple_lr:
    model = MultimodalSimCLR_MultipleLR(hparams)
  else:
    model = MultimodalSimCLR(hparams)

  if hparams.watch_weights:
    wandb_logger.watch(model, log='all', log_freq=10)
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor='multimodal.val.loss', mode='min', filename='checkpoint_best_loss', dirpath=logdir))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)

  return version