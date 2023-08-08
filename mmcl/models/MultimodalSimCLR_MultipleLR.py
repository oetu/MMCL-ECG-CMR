from typing import List, Tuple, Dict
import torch
from torch import nn
import torchvision
import torchmetrics
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightly.models.modules import SimCLRProjectionHead
from utils.ntx_ent_loss_custom import NTXentLoss
from utils.clip_loss import CLIPLoss
from utils.supcon_loss_custom import SupConLoss
from utils.supcon_loss_clip import SupConLossCLIP
from utils.kpositive_loss_clip import KPositiveLossCLIP
from utils.remove_fn_loss import RemoveFNLoss

from models.LinearClassifier import LinearClassifier


class MultimodalSimCLR_MultipleLR(pl.LightningModule):
  """
  Lightning module for multimodal SimCLR.

  Alternates training between contrastive model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # Imaging
    if hparams.model == 'resnet18':
      if hparams.load_imagenet_weights:
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
      else:
        resnet = torchvision.models.resnet18()
      pooled_dim = 512
    if hparams.model == 'resnet50':
      if hparams.load_imagenet_weights:
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
      else:
        resnet = torchvision.models.resnet50()
      pooled_dim = 2048
    self.encoder_imaging = nn.Sequential(*list(resnet.children())[:-1])
    if self.hparams.use_projection_head:
      self.projection_head_imaging = SimCLRProjectionHead(pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim)
    else:
      self.projection_head_imaging = nn.Identity()

    if self.hparams.imaging_pretrain_checkpoint:
      loaded_chkpt = torch.load(self.hparams.imaging_pretrain_checkpoint)
      state_dict = loaded_chkpt['state_dict']
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith('encoder_imaging.'):
          state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
      _ = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=True)
      print("Loaded imaging weights")
      if self.hparams.pretrained_imaging_strategy == 'frozen':
        for _, param in self.encoder_imaging.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
        assert len(parameters)==0
    
    # Tabular
    self.encoder_projector_tabular = EncoderProjector(hparams)

    # Multimodal
    if self.hparams.loss.lower() == 'remove_fn':
      self.criterion_train = RemoveFNLoss(temperature=self.hparams.temperature, cosine_similarity_matrix_path=self.hparams.train_similarity_matrix, threshold=self.hparams.threshold)
      self.criterion_val = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'supcon':
      self.criterion_train = SupConLossCLIP(temperature=self.hparams.temperature, contrast_mode='all', cosine_similarity_matrix_path=self.hparams.train_similarity_matrix, threshold=self.hparams.threshold)
      self.criterion_val = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'kpositive':
      self.criterion_train = KPositiveLossCLIP(temperature=self.hparams.temperature, k=6, cosine_similarity_matrix_path=self.hparams.train_similarity_matrix, threshold=self.hparams.threshold)
      self.criterion_val = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    elif self.hparams.loss.lower() == 'clip':
      self.criterion_train = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
      self.criterion_val = self.criterion_train
    elif self.hparams.loss.lower() == 'ntxent':  
      self.criterion_train = NTXentLoss(self.hparams.temperature, cosine_similarity_matrix_path=self.hparams.train_similarity_matrix, threshold=self.hparams.threshold, similarity_divisor=self.hparams.similarity_divisor)
      self.criterion_val = self.criterion_train
    else:
      raise ValueError('The only implemented losses currently are CLIP, NTXent, supcon, and remove_fn')

    # Defines weights to be used for the classifier in case of imbalanced data
    if not self.hparams.weights:
      self.hparams.weights = [1.0 for _ in range(self.hparams.num_classes)]
    self.weights = torch.tensor(self.hparams.weights)

    # Classifier
    self.classifier = LinearClassifier(in_size=pooled_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat)
    self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    self.top1_acc_train = torchmetrics.Accuracy(top_k=1)
    self.top1_acc_val = torchmetrics.Accuracy(top_k=1)

    self.top5_acc_train = torchmetrics.Accuracy(top_k=5)
    self.top5_acc_val = torchmetrics.Accuracy(top_k=5)

    self.f1_train = torchmetrics.F1Score(average=None, num_classes=hparams.num_classes)
    self.f1_val = torchmetrics.F1Score(average=None, num_classes=hparams.num_classes)

    self.classifier_acc_train = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')
    self.classifier_acc_val = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')

    self.classifier_auc_train = torchmetrics.AUROC(num_classes = self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(num_classes = self.hparams.num_classes)

    print(f'Tabular model, multimodal: {self.encoder_projector_tabular}')
    print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projection_head_imaging}')

    self.automatic_optimization = False

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of imaging data.
    """
    x = self.encoder_imaging(x).flatten(start_dim=1)
    z = self.projection_head_imaging(x)
    return z

  def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of tabular data.
    """
    z = self.encoder_projector_tabular(x)
    return z

  def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    im_views, tab_views, y, original_im, indices = batch
    
    opt_im, opt_tab, opt_clas = self.optimizers()

    # Train contrastive model
    if self.hparams.view == 'original':
      im = original_im
      table = tab_views[0]
    elif self.hparams.view == 'augmented':
      im = im_views[1]
      table = tab_views[1]
    else:
      raise ValueError('Valid views are "original" and "augmented"')
    z0 = self.forward_imaging(im) # Using second view here for chance of untransformed image
    z1 = self.forward_tabular(table) # Using the corrupted tab view here
    loss, logits, labels = self.criterion_train(z0, z1, indices)

    self.top1_acc_train(logits, labels)
    self.top5_acc_train(logits, labels)
    
    self.log("multimodal.train.loss", loss, on_epoch=True, on_step=False)
    self.log("multimodal.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
    self.log("multimodal.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)

    opt_im.zero_grad()
    opt_tab.zero_grad()
    self.manual_backward(loss)
    opt_im.step()
    opt_tab.step()
  
    # Train classifier
    embedding = torch.squeeze(self.encoder_imaging(original_im))
    y_hat = self.classifier(embedding)
    loss = self.classifier_criterion(y_hat, y)

    self.f1_train(y_hat, y)
    self.classifier_acc_train(y_hat, y)
    self.classifier_auc_train(y_hat, y)

    self.log('classifier.train.loss', loss, on_epoch=True, on_step=False)
    self.log('classifier.train.f1', self.f1_train[1], on_epoch=True, on_step=False, metric_attribute=self.f1_train)
    self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
    self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)

    opt_clas.zero_grad()
    self.manual_backward(loss)
    opt_clas.step()

  def training_epoch_end(self, _) -> None:
    sch_im, sch_tab = self.lr_schedulers()
    sch_im.step()
    sch_tab.step()


  def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    im_views, tab_views, y, original_im, indices = batch
    
    # Validate contrastive model
    z0 = self.forward_imaging(original_im)
    z1 = self.forward_tabular(tab_views[0]) # Using the uncorrupted tab view here
    loss, logits, labels = self.criterion_val(z0, z1, indices)

    self.top1_acc_val(logits, labels)
    self.top5_acc_val(logits, labels)

    self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
    self.log("multimodal.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log("multimodal.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)

    # Validate classifier
    self.classifier.eval()
    embedding = torch.squeeze(self.encoder_imaging(original_im))
    y_hat = self.classifier(embedding)
    loss = self.classifier_criterion(y_hat, y)

    self.f1_val(y_hat, y)
    self.classifier_acc_val(y_hat, y)
    self.classifier_auc_val(y_hat, y)

    self.log('classifier.val.loss', loss, on_epoch=True, on_step=False)
    self.log('classifier.val.f1', self.f1_val[1], on_epoch=True, on_step=False, metric_attribute=self.f1_val)
    self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
    self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)
    self.classifier.train()

    return im_views[1]

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
    optimizer_imaging = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projection_head_imaging.parameters()}
      ], lr=self.hparams.lr_imaging, weight_decay=self.hparams.weight_decay)
    optimizer_tabular = torch.optim.Adam(self.encoder_projector_tabular.parameters(), lr=self.hparams.lr_tabular, weight_decay=self.hparams.weight_decay)
    optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr_classifier, weight_decay=self.hparams.weight_decay_classifier)
    
    if self.hparams.scheduler == 'cosine':
      scheduler_imaging = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_imaging, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
      scheduler_tabular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tabular, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler_imaging = LinearWarmupCosineAnnealingLR(optimizer_imaging, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
      scheduler_tabular = LinearWarmupCosineAnnealingLR(optimizer_tabular, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    
    return [optimizer_imaging, optimizer_tabular, optimizer_classifier], [scheduler_imaging, scheduler_tabular]
    