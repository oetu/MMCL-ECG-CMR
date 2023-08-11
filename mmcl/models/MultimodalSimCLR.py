import os
import sys

from typing import List, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np

from timm.models.layers import trunc_normal_
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightly.models.modules import SimCLRProjectionHead
from utils.ntx_ent_loss_custom import NTXentLoss
from utils.clip_loss import CLIPLoss
from utils.plot_localization import plot_ecg_localization, plot_ecg_attention, plot_image_localization, plot_pairwise_localization
from utils.supcon_loss_custom import SupConLoss
from utils.supcon_loss_clip import SupConLossCLIP
from utils.kpositive_loss_clip import KPositiveLossCLIP
from utils.remove_fn_loss import RemoveFNLoss
import utils.pos_embed as pos_embed

from models.LinearClassifier import LinearClassifier
import models.ECGEncoder as ECGEncoder
import models.UNetEncoder as UNetEncoder


class MultimodalSimCLR(pl.LightningModule):
  """
  Lightning module for multimodal SimCLR.

  Alternates training between contrastive model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # path where to store embeddings
    self.embeddings_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'runs', 'multimodal', f'version_{hparams.version}', 'embeddings')
    if hparams.save_embeddings and not os.path.exists(self.embeddings_path):
        os.makedirs(self.embeddings_path)

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
    self.projection_head_imaging = SimCLRProjectionHead(pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

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
    self.projection_head_ecg = SimCLRProjectionHead(self.encoder_ecg.embed_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

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

    # Multimodal
    nclasses = hparams.batch_size
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
      self.criterion_train = NTXentLoss(self.hparams.temperature)
      self.criterion_val = self.criterion_train
      nclasses = hparams.batch_size*2-1
    else:
      raise ValueError('The only implemented losses currently are CLIP, NTXent, supcon, and remove_fn')

    # Defines weights to be used for the classifier in case of imbalanced data
    if not self.hparams.weights:
      self.hparams.weights = [1.0 for _ in range(self.hparams.num_classes)]
    self.weights = torch.tensor(self.hparams.weights)

    # Classifier
    if hparams.online_classifier == "image":
      self.classifier = LinearClassifier(in_size=pooled_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat) # image
    else:
      self.classifier = LinearClassifier(in_size=self.encoder_ecg.embed_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat) # ecg
    self.online_classifier = hparams.online_classifier

    self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    self.top1_acc_train = torchmetrics.Accuracy(top_k=1)
    self.top1_acc_val = torchmetrics.Accuracy(top_k=1)
    # self.top1_acc_train = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=nclasses)
    # self.top1_acc_val = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=nclasses)

    self.top5_acc_train = torchmetrics.Accuracy(top_k=5)
    self.top5_acc_val = torchmetrics.Accuracy(top_k=5)
    # self.top5_acc_train = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=nclasses)
    # self.top5_acc_val = torchmetrics.Accuracy(task="multiclass", top_k=5, num_classes=nclasses)

    self.f1_train = torchmetrics.F1Score(average=None, num_classes=hparams.num_classes)
    self.f1_val = torchmetrics.F1Score(average=None, num_classes=hparams.num_classes)
    # self.f1_train = torchmetrics.F1Score(task="binary", average=None, num_classes=hparams.num_classes)
    # self.f1_val = torchmetrics.F1Score(task="binary", average=None, num_classes=hparams.num_classes)

    self.classifier_acc_train = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')
    self.classifier_acc_val = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted')
    # self.classifier_acc_train = torchmetrics.Accuracy(task="binary", num_classes = self.hparams.num_classes, average = 'weighted')
    # self.classifier_acc_val = torchmetrics.Accuracy(task="binary", num_classes = self.hparams.num_classes, average = 'weighted')

    self.classifier_auc_train = torchmetrics.AUROC(num_classes = self.hparams.num_classes)
    self.classifier_auc_val = torchmetrics.AUROC(num_classes = self.hparams.num_classes)
    # self.classifier_auc_train = torchmetrics.AUROC(task="binary", num_classes = self.hparams.num_classes)
    # self.classifier_auc_val = torchmetrics.AUROC(task="binary", num_classes = self.hparams.num_classes)

    print(f'ECG model, multimodal: {self.encoder_ecg}\n{self.projection_head_ecg}')
    print(f'Imaging model, multimodal: {self.encoder_imaging}\n{self.projection_head_imaging}')

  def forward_imaging(self, x: torch.Tensor, localized=False) -> torch.Tensor:
    """
    Generates projection of imaging data.

    :input:
    x (B, C, H_img, W_img)

    :return: 
      localized == False => (B, d)
      localized == True => (B, H', W', d)
    """
    if localized:
      encoder_imaging_localized = nn.Sequential(*list(self.encoder_imaging.children())[:-1]) # remove the global avg pool layer
      # (B, d_encoder, H, W)
      x = encoder_imaging_localized(x)
      if self.hparams.upsample_factor_img != 1:
        # (B, d_encoder, H', W')
        x = F.interpolate(x, scale_factor=self.hparams.upsample_factor_img, mode='bilinear')
      B, d, H, W = x.shape
      # (B*H'*W', d_encoder)
      x = x.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
      z = self.projection_head_imaging(x)
      # (B, H', W', d_prj)
      z = z.view(B, H, W, -1)
    else:
      # (B, d_encoder)
      x = self.encoder_imaging(x).flatten(start_dim=1)
      # (B, d_prj)
      z = self.projection_head_imaging(x)
    return z

  def forward_ecg(self, x: torch.Tensor, localized=False) -> torch.Tensor:
    """
    Generates projection of ecg data.

    :input:
    x (B, C, C_sig, T_sig)

    :return: 
      localized == False => (B, d)
      localized == True => (B, N', d)
    """
    if localized:
      # (B, N, d_encoder)
      x = self.encoder_ecg.forward_features(x, localized=True)
      if self.hparams.upsample_factor_ecg != 1:
        # (B, d_encoder, N)
        x = x.permute(0, 2, 1)
        # (B, d_encoder, N')
        x = F.interpolate(x, scale_factor=self.hparams.upsample_factor_ecg, mode='linear')
        # (B, N', d_encoder)
        x = x.permute(0, 2, 1)
      B, N, d = x.shape
      # (B*N', d_encoder)
      x = x.flatten(start_dim=0, end_dim=1)
      # (B*N', d_prj)
      z = self.projection_head_ecg(x)
      # (B, N', d_prj)
      z = z.view(B, N, -1)
    else:
      # (B, d_encoder)
      x = self.encoder_ecg.forward_features(x).flatten(start_dim=1)
      # (B, d_prj)
      z = self.projection_head_ecg(x)
    return z

  def _attention_forward_wrapper(self, attn_obj):
    """
    Modified version of def forward() of class Attention() in timm.models.vision_transformer
    """
    def my_forward(x):
        B, N, C = x.shape # C = embed_dim
        # (3, B, Heads, N, head_dim)
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        # (B, Heads, N, N)
        attn_obj.attn_map = attn # this was added 

        # (B, N, Heads*head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

  def plot_localization(self, original_im, original_ecg, plot_pairwise=False):
    # (B, d)
    z_im = self.forward_imaging(original_im)
    # (B, H', W', d)
    z_local_im = self.forward_imaging(original_im, localized=True)
    B, H, W, d = z_local_im.shape
    B, _, C_sig, T_sig = original_ecg.shape
    # (B, d)
    z_ecg = self.forward_ecg(original_ecg)
    # (B, C_sig*N'_(C_sig), d) with N'_(C_sig) = #patches per C_sig 
    z_local_ecg = self.forward_ecg(original_ecg, localized=True)

    # read out attention map
    # (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    attention_map_ecg = self.encoder_ecg.blocks[-1].attn.attn_map

    # Normalization to facilitate calculation of cosine similarity
    z_im = F.normalize(z_im, dim=-1)
    z_local_im = F.normalize(z_local_im, dim=-1)
    # (B, H'*W', d)
    z_local_im_flattened = z_local_im.flatten(start_dim=1, end_dim=2)
    z_ecg = F.normalize(z_ecg, dim=-1)
    z_local_ecg = F.normalize(z_local_ecg, dim=-1)

    # (B, C_sig*N'_(C_sig)) = (B, C_sig*N'_(C_sig), d) x (B, d, 1)
    importance_ecg = torch.bmm(z_local_ecg, z_im.unsqueeze(-1)).squeeze(-1) / self.hparams.temp
    if self.hparams.use_softmax:
      importance_ecg = importance_ecg.softmax(dim=1)
    # (B, C_sig, N'_(C_sig))
    importance_ecg = importance_ecg.view(B, C_sig, -1)

    # (B, H'*W') = (B, H'*W', d) x (B, d, 1)
    importance_im = torch.bmm(z_local_im_flattened, z_ecg.unsqueeze(-1)).squeeze(-1) / self.hparams.temp
    if self.hparams.use_softmax:
      importance_im = importance_im.softmax(dim=1)
    # (B, H', W')
    importance_im = importance_im.view(B, H, W)

    # (B, C_sig*N'_(C_sig), H'*W') = (B, C_sig*N'_(C_sig), d) x (B, d, H'*W')
    importance_pairwise = torch.bmm(z_local_ecg, z_local_im_flattened.transpose(1, 2)) / self.hparams.temp
    if self.hparams.use_softmax:
      importance_pairwise = importance_pairwise.softmax(dim=-1)
    # (B, C_sig, N'_(C_sig), H', W')
    importance_pairwise = importance_pairwise.view(B, C_sig, -1, H, W)
    # (B, N'_(C_sig), H', W')
    importance_pairwise = importance_pairwise.mean(1)

    idx = int(torch.rand(1).item()*32)
    plot_image_localization(original_im, importance_im, idx)
    plot_ecg_localization(original_ecg, importance_ecg, idx)
    plot_ecg_attention(original_ecg, attention_map_ecg, idx)
    if plot_pairwise:
      plot_pairwise_localization(original_im, original_ecg, importance_pairwise, idx)


  def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _, optimizer_idx: int) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    augmented_image, augmented_ecg, y, original_im, original_ecg, indices = batch

    # Train contrastive model
    if optimizer_idx == 0:
      z0 = self.forward_imaging(augmented_image)
      z1 = self.forward_ecg(augmented_ecg)
      loss, logits, labels = self.criterion_train(z0, z1, indices)

      self.top1_acc_train(logits, labels)
      if len(logits)>5:
        self.top5_acc_train(logits, labels)
      
      self.log("multimodal.train.loss", loss, on_epoch=True, on_step=False)
      self.log("multimodal.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
      self.log("multimodal.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)
    
    # Train classifier
    if optimizer_idx == 1:
      if self.online_classifier == "image":
        embedding = torch.squeeze(self.encoder_imaging(original_im)) # image
      else:
        embedding = torch.squeeze(self.encoder_ecg.forward_features(original_ecg)) # ecg
      y_hat = self.classifier(embedding)
      loss = self.classifier_criterion(y_hat, y)

      # self.f1_train(torch.argmax(y_hat, dim=1), y)
      # self.classifier_acc_train(torch.argmax(y_hat, dim=1), y)
      self.f1_train(y_hat, y)
      self.classifier_acc_train(y_hat, y)
      self.classifier_auc_train(y_hat, y)

      self.log('classifier.train.loss', loss, on_epoch=True, on_step=False)
      self.log('classifier.train.f1', self.f1_train[1], on_epoch=True, on_step=False, metric_attribute=self.f1_train)
      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)

    return loss

  def validation_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    augmented_image, augmented_ecg, y, original_im, original_ecg, indices = batch
    
    # Validate contrastive model
    z0 = self.forward_imaging(original_im)
    z1 = self.forward_ecg(original_ecg)
    if self.hparams.save_embeddings:
      torch.save(z0.detach().cpu(), os.path.join(self.embeddings_path, f"embeddings_imaging_step{self.global_step}_batch{batch_idx}.pt"))
      torch.save(z1.detach().cpu(), os.path.join(self.embeddings_path, f"embeddings_signal_step{self.global_step}_batch{batch_idx}.pt"))
    loss, logits, labels = self.criterion_val(z0, z1, indices)

    self.top1_acc_val(logits, labels)
    if len(logits)>5:
      self.top5_acc_val(logits, labels)

    self.log("multimodal.val.loss", loss, on_epoch=True, on_step=False)
    self.log("multimodal.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log("multimodal.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)

    # Validate classifier
    self.classifier.eval()
    if self.online_classifier == "image":
      embedding = torch.squeeze(self.encoder_imaging(original_im)) # image
    else:
      embedding = torch.squeeze(self.encoder_ecg.forward_features(original_ecg)) # ecg
    y_hat = self.classifier(embedding)
    loss = self.classifier_criterion(y_hat, y)

    self.f1_val(y_hat, y)
    self.classifier_acc_val(y_hat, y)
    # self.f1_val(torch.argmax(y_hat, dim=1), y)
    # self.classifier_acc_val(torch.argmax(y_hat, dim=1), y)
    self.classifier_auc_val(y_hat, y)

    self.log('classifier.val.loss', loss, on_epoch=True, on_step=False)
    self.log('classifier.val.f1', self.f1_val[1], on_epoch=True, on_step=False, metric_attribute=self.f1_val)
    self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
    self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)
    self.classifier.train()

    # Plot localization
    if self.hparams.plot_localization and batch_idx == 0:
      self.plot_localization(original_im, original_ecg, plot_pairwise=self.hparams.plot_localization_pairwise)

    return original_im, original_ecg, augmented_image, augmented_ecg

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step
    """
    outs = validation_step_outputs[0]
    outs = [x.cpu().detach().numpy() for x in outs]
    if self.hparams.log_images and self.current_epoch < 5:
      indx = np.random.randint(0, outs[0].shape[0])
      self.logger.log_image(key="Original vs Augmented Image", images=[outs[0][indx].transpose(1,2,0), outs[2][indx].transpose(1,2,0)])
      plt.close('all')
      plt.subplot(211)
      plt.plot(range(0, int(outs[1].shape[-1]/10), 1), outs[1][indx, 0, 0, ::10])
      plt.subplot(212)
      plt.plot(range(0, int(outs[1].shape[-1]/10), 1), outs[3][indx, 0, 0, ::10])
      plt.tight_layout()
      self.logger.log_image(key='Original vs Augmented ECG',images=[plt.gcf()])

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.Adam(
      [
        {'params': self.encoder_imaging.parameters()}, 
        {'params': self.projection_head_imaging.parameters()},
        {'params': self.encoder_ecg.parameters()},
        {'params': self.projection_head_ecg.parameters()},
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.lr_classifier, weight_decay=self.hparams.weight_decay_classifier)
    
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
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