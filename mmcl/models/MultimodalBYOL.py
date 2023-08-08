from typing import List, Tuple, Dict, Any
import copy 
import torch
from torch import nn
import torchvision
import torchmetrics
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.loss import NegativeCosineSimilarity
from sklearn.linear_model import LogisticRegression

import models.ECGEncoder as ECGEncoder
import utils.pos_embed as pos_embed

from utils.utils import calc_logits_labels


class MultimodalBYOL(pl.LightningModule):
  """
  Lightning module for multimodal BYOL.

  Alternates training between BYOL model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # Imaging
    if hparams.model == 'resnet18':
      resnet = torchvision.models.resnet18()
      pooled_dim = 512
    if hparams.model == 'resnet50':
      resnet = torchvision.models.resnet50()
      pooled_dim = 2048

    self.encoder_imaging = nn.Sequential(*list(resnet.children())[:-1])
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

    self.encoder_imaging_momentum = copy.deepcopy(self.encoder_imaging)

    self.projection_head_imaging = BYOLProjectionHead(input_dim = pooled_dim, hidden_dim = self.hparams.embedding_dim, output_dim = self.hparams.projection_dim)
    self.projection_head_imaging_momentum = copy.deepcopy(self.projection_head_imaging)

    self.prediction_head_imaging = BYOLPredictionHead(input_dim = self.hparams.projection_dim, hidden_dim = self.hparams.embedding_dim, output_dim = self.hparams.projection_dim)
    
    # ECG
    if hparams.attention_pool:
      hparams.global_pool = "attention_pool"
    self.encoder_ecg = ECGEncoder.__dict__[hparams.ecg_model](
        img_size=hparams.input_size,
        patch_size=hparams.patch_size,
        num_classes=hparams.num_classes,
        drop_path_rate=hparams.drop_path,
        global_pool=hparams.global_pool)
    self.encoder_ecg.blocks[-1].attn.forward = self._attention_forward_wrapper(self.encoder_ecg.blocks[-1].attn) # required to read out the attention map of the last layer
    
    if self.hparams.ecg_pretrain_checkpoint:
      checkpoint = torch.load(hparams.ecg_pretrain_checkpoint)
      print("Load pre-trained checkpoint from: %s" % hparams.ecg_pretrain_checkpoint)
      checkpoint_model = checkpoint['model']
      state_dict = self.encoder_ecg.state_dict()
      for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
          print(f"Removing key {k} from pretrained checkpoint")
          del checkpoint_model[k]

      # interpolate position embedding
      pos_embed.interpolate_pos_embed(self.encoder_ecg, checkpoint_model)

      # load pre-trained model
      msg = self.encoder_ecg.load_state_dict(checkpoint_model, strict=False)
      print(msg)

    self.encoder_ecg_momentum = copy.deepcopy(self.encoder_ecg)

    self.projection_head_ecg = BYOLProjectionHead(input_dim = pooled_dim, hidden_dim = self.hparams.embedding_dim, output_dim = self.hparams.projection_dim)
    self.projection_head_ecg_momentum = copy.deepcopy(self.projection_head_ecg)

    self.prediction_head_ecg = BYOLPredictionHead(input_dim = self.hparams.projection_dim, hidden_dim = self.hparams.embedding_dim, output_dim = self.hparams.projection_dim)

    # BYOL
    deactivate_requires_grad(self.encoder_imaging_momentum)
    deactivate_requires_grad(self.projection_head_imaging_momentum)
    deactivate_requires_grad(self.encoder_ecg_momentum)
    deactivate_requires_grad(self.projection_head_ecg_momentum)

    # Multimodal
    self.criterion_train = NegativeCosineSimilarity()
    self.criterion_val = NegativeCosineSimilarity()
    
    # Defines weights to be used for the classifier in case of imbalanced data
    if not self.hparams.weights:
      self.hparams.weights = [1.0 for _ in range(self.hparams.num_classes)]
    self.weights = torch.tensor(self.hparams.weights)

    # Classifier
    self.estimator = None

    nclasses = self.hparams.batch_size
    self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses)
    self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses)

    self.top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses)
    self.top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

    self.classifier_acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

    self.classifier_auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

    print(f'ECG model, multimodal: {self.encoder_ecg}\n{self.projection_head_ecg}')
    print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projection_head_imaging}')

  def forward_imaging_momentum(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of imaging data.
    """
    x = self.encoder_imaging_momentum(x).flatten(start_dim=1)
    z = self.projection_head_imaging_momentum(x)
    z = z.detach()

    return z

  def forward_ecg_momentum(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of ecg data.
    """
    x = self.encoder_ecg_momentum(x).flatten(start_dim=1)
    z = self.projection_head_ecg_momentum(x)
    z = z.detach()

    return z

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates BYOL predction of imaging data.
    """
    x = self.encoder_imaging(x).flatten(start_dim=1)
    z = self.projection_head_imaging(x)
    p = self.prediction_head_imaging(z)

    return p, x

  def forward_ecg(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates BYOL prediction of ecg data.
    """
    x = self.encoder_ecg(x).flatten(start_dim=1)
    z = self.projection_head_ecg(x)
    p = self.prediction_head_ecg(z)
    
    return p, x

  def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    im_views, tab_views, y, original_im, indices = batch

    # Train contrastive model
    im = im_views[1]
    table = tab_views[1]

    update_momentum(self.encoder_imaging, self.encoder_imaging_momentum, self.hparams.momentum)
    update_momentum(self.projection_head_imaging, self.projection_head_imaging_momentum, self.hparams.momentum)
    update_momentum(self.encoder_ecg, self.encoder_ecg_momentum, self.hparams.momentum)
    update_momentum(self.projection_head_ecg, self.projection_head_ecg_momentum, self.hparams.momentum)

    pred_im, emb_im = self.forward_imaging(im)
    pred_tab, emb_tab = self.forward_ecg(table)
    proj_mom_im = self.forward_imaging_momentum(im)
    proj_mom_tab = self.forward_ecg_momentum(table)

    loss_1 = self.criterion_train(pred_im, proj_mom_tab)
    loss_2 = self.criterion_train(pred_tab, proj_mom_im)

    loss = (loss_1 + loss_2)*0.5

    logits_1, labels_1 = calc_logits_labels(pred_im, proj_mom_tab)
    logits_2, labels_2 = calc_logits_labels(pred_tab, proj_mom_im)

    if len(im_views[0])==self.hparams.batch_size:
      self.top1_acc_train(logits_1, labels_1)
      self.top1_acc_train(logits_2, labels_2)

      self.top5_acc_train(logits_1, labels_1)
      self.top5_acc_train(logits_2, labels_2)
      
      self.log("multimodal.train.loss", loss, on_epoch=True, on_step=False)
      self.log("multimodal.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
      self.log("multimodal.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)
    
    return {'loss':loss, 'embeddings': emb_im, 'labels': y}

  def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
    """
    Train classifier
    """
    if self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(train_step_outputs)
    
      self.estimator = LogisticRegression(class_weight='balanced', max_iter=5000).fit(embeddings, labels)
      preds, probs = self.predict_live_estimator(embeddings)

      self.classifier_acc_train(preds, labels)
      self.classifier_auc_train(probs, labels)

      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)


  def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    im_views, tab_views, y, original_im, indices = batch
    
    # Validate contrastive model
    im = im_views[0]
    table = tab_views[0]
    pred_im, emb_im = self.forward_imaging(im)
    pred_tab, emb_tab = self.forward_ecg(table)
    proj_mom_im = self.forward_imaging_momentum(im)
    proj_mom_tab = self.forward_ecg_momentum(table)
    
    loss_1 = self.criterion_train(pred_im, proj_mom_tab)
    loss_2 = self.criterion_train(pred_tab, proj_mom_im)

    loss = (loss_1 + loss_2)*0.5

    logits_1, labels_1 = calc_logits_labels(pred_im, proj_mom_tab)
    logits_2, labels_2 = calc_logits_labels(pred_tab, proj_mom_im)

    if len(im_views[1])==self.hparams.batch_size:
      self.top1_acc_val(logits_1, labels_1)
      self.top1_acc_val(logits_2, labels_2)

      self.top5_acc_val(logits_1, labels_1)
      self.top5_acc_val(logits_2, labels_2)

      self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
      self.log("multimodal.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
      self.log("multimodal.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)


    return {'sample_augmentation': im_views[1], 'embeddings': emb_im, 'labels': y}

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step
    """
    if self.hparams.log_images:
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

    # Validate classifier
    if not self.estimator is None and self.current_epoch % self.hparams.classifier_freq == 0:
      embeddings, labels = self.stack_outputs(validation_step_outputs)
      
      preds, probs = self.predict_live_estimator(embeddings)
      
      self.classifier_acc_val(preds, labels)
      self.classifier_auc_val(probs, labels)

      #self.log('classifier.val.f1', self.classifier_f1_val[1], metric_attribute=self.classifier_f1_val)
      self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
      self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)

  def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack outputs from multiple steps
    """
    labels = outputs[0]['labels']
    embeddings = outputs[0]['embeddings']
    for i in range(1, len(outputs)):
      labels = torch.cat((labels, outputs[i]['labels']), dim=0)
      embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

    embeddings = embeddings.detach().cpu()
    labels = labels.cpu()

    return embeddings, labels

  def predict_live_estimator(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict using live estimator
    """
    preds = self.estimator.predict(embeddings)
    probs = self.estimator.predict_proba(embeddings)

    preds = torch.tensor(preds)
    probs = torch.tensor(probs)
    
    # Only need probs for positive class in binary case
    if self.hparams.num_classes == 2:
      probs = probs[:,1]

    return preds, probs

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projection_head_imaging.parameters()},
        {'params': self.prediction_head_imaging.parameters()},
        {'params': self.encoder_ecg.parameters()},
        {'params': self.projection_head_ecg.parameters()},
        {'params': self.prediction_head_ecg.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      }
    )