import os 
import sys

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from datasets.EvalImageDataset import EvalImageDataset
from datasets.EvalECGDataset import EvalECGDataset
from datasets.EvalImagingAndTabularDataset import EvalImagingAndTabularDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, get_next_version


def evaluate(hparams, wandb_logger):
  """
  Evaluates trained contrastive models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.datatype == 'imaging':
    train_dataset = EvalImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams.checkpoint, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task)
    val_dataset = EvalImageDataset(hparams.data_val_eval_imaging, hparams.labels_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams.checkpoint, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task)
  elif hparams.datatype == 'ecg' or hparams.datatype == 'multimodal':
    train_dataset = EvalECGDataset(hparams.data_train_eval_ecg, hparams.labels_train_eval_ecg, hparams.eval_train_augment_rate, train=True, args=hparams)
    val_dataset = EvalECGDataset(hparams.data_val_eval_ecg, hparams.labels_val_eval_ecg, hparams.eval_train_augment_rate, train=False, args=hparams)
  elif hparams.datatype == 'imaging_and_tabular':
    transform = grab_hard_eval_image_augmentations(grab_arg_from_checkpoint(hparams.checkpoint, 'img_size'), target=hparams.target)
    train_dataset = EvalImagingAndTabularDataset(
      hparams.data_train_eval_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, train=True
    )
    val_dataset = EvalImagingAndTabularDataset(
      hparams.data_val_eval_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
      hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_val_eval_imaging, hparams.img_size, hparams.live_loading, train=False
    )
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception('argument dataset must be set to imaging, tabular, multimodal or imaging_and_tabular')
  
  drop = ((len(train_dataset)%hparams.batch_size)==1)

  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size, 
    pin_memory=True, shuffle=True, persistent_workers=True, drop_last=drop)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,
    pin_memory=True, shuffle=False, persistent_workers=True)

  basepath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'runs', 'eval')
  version = str(get_next_version(basepath))
  logdir = os.path.join(basepath,f'version_{version}')
  os.makedirs(logdir,exist_ok=True)

  hparams.version = version

  if hparams.task == 'regression':
    model = Evaluator_Regression(hparams)
    metric = 'mse'
    mode = 'min'
  else:
    model = Evaluator(hparams)
    metric = 'acc' if hparams.target == 'dvm' else 'auc'
    mode = 'max'
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'eval.val.{metric}', mode=mode, filename=f'checkpoint_best_{metric}', dirpath=logdir))
  callbacks.append(EarlyStopping(monitor=f'eval.val.{metric}', min_delta=0.0002, patience=int(10*(1/hparams.val_check_interval)), verbose=False, mode=mode))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, val_check_interval=hparams.val_check_interval)

  trainer.fit(model, train_loader, val_loader)
  if hparams.use_wandb:
    wandb_logger.log_metrics({f'best.val.{metric}': model.best_val_score})

  if hparams.test_and_eval:
    if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
      test_dataset = EvalImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams.checkpoint, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task)
      
      hparams.transform_test = test_dataset.transform_val.__repr__()
    elif hparams.datatype == 'tabular':
      test_dataset = EvalTabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
      hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == 'imaging_and_tabular':
      test_dataset = EvalImagingAndTabularDataset(
        hparams.data_test_eval_imaging, hparams.delete_segmentation, transform, 0, 
        hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False)
      hparams.input_size = test_dataset.get_input_size()
    else:
      raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
      pin_memory=True, shuffle=False, persistent_workers=True, drop_last=drop)

    model.freeze()

    trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{metric}.ckpt'))
    #trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,'checkpoint_best_loss.ckpt'))