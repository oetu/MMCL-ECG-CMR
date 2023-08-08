from typing import List, Tuple, Dict
import torch
from torch import nn
import torchvision
import torchmetrics
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightly.models.modules import SimCLRProjectionHead
from utils.ntx_ent_loss_custom import NTXentLoss

from models.LinearClassifier import LinearClassifier
import models.UNetEncoder as UNetEncoder

class SimCLR(pl.LightningModule):
  """
  Lightning module for imaging SimCLR.

  Alternates training between contrastive model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # Imaging
    if hparams.model == 'resnet18':
      resnet = torchvision.models.resnet18()
      pooled_dim = 512
      self.encoder_imaging = nn.Sequential(*list(resnet.children())[:-1])
    if hparams.model == 'resnet50':
      resnet = torchvision.models.resnet50()
      pooled_dim = 2048
      self.encoder_imaging = nn.Sequential(*list(resnet.children())[:-1])
    if hparams.model == 'unet':
      resnet = UNetEncoder.UNetEncoder(ndim=2, enc_channels=(16, 32, 32, 32, 32))
      pooled_dim = 32
      self.encoder_imaging = nn.Sequential(*list(resnet.children()), nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    self.projection_head = SimCLRProjectionHead(pooled_dim, pooled_dim, self.hparams.projection_dim)
    self.criterion_train = NTXentLoss(temperature=self.hparams.temperature)
    self.criterion_val = NTXentLoss(temperature=self.hparams.temperature)
    
    # Defines weights to be used for the classifier in case of imbalanced data
    if not self.hparams.weights:
      self.hparams.weights = [1.0 for i in range(self.hparams.num_classes)]
    self.weights = torch.tensor(self.hparams.weights)

    # Classifier
    self.classifier = LinearClassifier(in_size=pooled_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat)
    self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    self.top1_acc_train = torchmetrics.Accuracy(top_k=1)
    self.top1_acc_val = torchmetrics.Accuracy(top_k=1)

    self.top5_acc_train = torchmetrics.Accuracy(top_k=5)
    self.top5_acc_val = torchmetrics.Accuracy(top_k=5)

    self.f1_train = torchmetrics.F1Score()
    self.f1_val = torchmetrics.F1Score()

    self.classifier_acc_train = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')
    self.classifier_acc_val = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')

    print(self.encoder_imaging)
    print(self.projection_head)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of data.
    """
    x = self.encoder_imaging(x).flatten(start_dim=1)
    z = self.projection_head(x)
    return z

  def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _, optimizer_idx: int) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    x0, x1, y, indices = batch

    # Train contrastive model
    if optimizer_idx == 0:
      z0 = self.forward(x0)
      z1 = self.forward(x1)
      loss, logits, labels = self.criterion_train(z0, z1, indices)

      self.top1_acc_train(logits, labels)
      self.top5_acc_train(logits, labels)
      
      self.log("imaging.train.loss", loss, on_epoch=True, on_step=False)
      self.log("imaging.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
      self.log("imaging.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)
    
    # Train classifier
    if optimizer_idx == 1:
      embedding = torch.squeeze(self.encoder_imaging(x0))
      y_hat = self.classifier(embedding)
      loss = self.classifier_criterion(y_hat, y)

      self.f1_train(y_hat, y)
      self.classifier_acc_train(y_hat, y)

      self.log('classifier.train.loss', loss, on_epoch=True, on_step=False)
      self.log('classifier.train.f1', self.f1_train, on_epoch=True, on_step=False)
      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)

    return loss


  def validation_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    x0, x1, y, indices = batch
    
    # Validate contrastive model
    z0 = self.forward(x0)
    z1 = self.forward(x1)
    loss, logits, labels = self.criterion_val(z0, z1, indices)

    self.top1_acc_val(logits, labels)
    self.top5_acc_val(logits, labels)

    self.log("imaging.val.loss", loss, on_epoch=True, on_step=False)
    self.log("imaging.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log("imaging.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)

    # Validate classifier
    self.classifier.eval()
    embedding = torch.squeeze(self.encoder_imaging(x0))
    y_hat = self.classifier(embedding)
    loss = self.classifier_criterion(y_hat, y)

    self.f1_val(y_hat, y)
    self.classifier_acc_val(y_hat, y)

    self.log('classifier.val.loss', loss, on_epoch=True, on_step=False)
    self.log('classifier.val.f1', self.f1_val, on_epoch=True, on_step=False)
    self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
    self.classifier.train()

    return x0

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step
    """
    if self.hparams.log_images:
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]])

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projection_head.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr_classifier, weight_decay=self.hparams.weight_decay_classifier)
    
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.anneal_max_epochs, eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs)
    classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, patience=int(20/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr_classifier*0.0001)
    
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      },
      { # Classifier
        "optimizer": classifier_optimizer
      }
    )