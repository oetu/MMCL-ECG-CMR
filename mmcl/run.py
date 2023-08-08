import os 
import sys
import time
import random

import hydra
from omegaconf import DictConfig, open_dict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from trainers.train_imaging import train_imaging
from trainers.train_ecg import train_ecg
from trainers.train_multimodal import train_multimodal
from trainers.evaluate import evaluate
from trainers.test import test
from trainers.generate_embeddings import generate_embeddings
from utils.utils import grab_arg_from_checkpoint, prepend_paths, chkpt_contains_arg

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  pl.seed_everything(args.seed)
  args = prepend_paths(args)
  time.sleep(random.randint(1,10)) # Prevents multiple runs getting the same version

  if args.resume_training:
    wandb_id = args.wandb_id
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    with open_dict(args):
      args.checkpoint = checkpoint
      args.resume_training = True
      if not wandb_id in args or not args.wandb_id:
        args.wandb_id = wandb_id
  
  if args.generate_embeddings:
    if not args.datatype:
      args.datatype = grab_arg_from_checkpoint(args.checkpoint, 'dataset')
    generate_embeddings(args)
    return
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  if args.use_wandb:
    if args.test or args.test_and_eval:
      wandb_logger = WandbLogger(project='MMCL Eval and Test', offline=args.offline)
    elif args.task == 'regression':
      wandb_logger = WandbLogger(project='MMCL Regression', offline=args.offline)
    elif args.resume_training and args.wandb_id:
      wandb_logger = WandbLogger(project='ecg_multimodal', save_dir=base_dir, offline=args.offline, id=args.wandb_id, resume='must')
    else:
      wandb_logger = WandbLogger(project='ecg_multimodal', save_dir=base_dir, offline=args.offline)
      args.wandb_id = wandb_logger.version
  else:
    wandb_logger = None

  checkpoints = []
  datasets = []

  if args.checkpoint and not args.resume_training:
    checkpoints.append(args.checkpoint)
    if not args.datatype:
      if chkpt_contains_arg(args.checkpoint, 'datatype'):
        args.datatype = grab_arg_from_checkpoint(args.checkpoint, 'datatype')
      else:
        args.datatype = grab_arg_from_checkpoint(args.checkpoint, 'dataset')
    datasets.append(args.datatype)

  if args.run_imaging:
    args.datatype = 'imaging'
    version = train_imaging(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'imaging', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('imaging')
  
  if args.run_ecg:
    args.datatype = 'ecg'
    version = train_ecg(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'ecg', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('ecg')

  if args.run_multimodal:
    args.datatype = 'multimodal'
    version = train_multimodal(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'multimodal', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('multimodal')
  
  # for i in range(len(checkpoints)):
  #   args.checkpoint = checkpoints[i]
  #   args.datatype = datasets[i]
  #   if args.test:
  #     test(args, wandb_logger)
  #   else:
  #     evaluate(args, wandb_logger)

if __name__ == "__main__":
  run()