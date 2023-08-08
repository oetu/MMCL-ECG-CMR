import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from lightly.models.modules import SimCLRProjectionHead

class ResnetEvalModel(nn.Module):
  """
  Evaluation model for imaging trained with ResNet backbone.
  """
  def __init__(self, args) -> None:
    super(ResnetEvalModel, self).__init__()

    self.keep_projector = args.keep_projector
    self.vec2vec = args.vec2vec

    # Load weights
    checkpoint = torch.load(args.checkpoint)
    original_args = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']

    # Load architecture
    if original_args['model'] == 'resnet18':
      model = models.resnet18(pretrained=False, num_classes=100)
      pooled_dim = 512
    elif original_args['model'] == 'resnet50':
      model = models.resnet50(pretrained=False, num_classes=100)
      pooled_dim = 2048
    else:
      raise Exception('Invalid architecture. Please select either resnet18 or resnet50.')

    self.backbone = nn.Sequential(*list(model.children())[:-1])

    # Remove prefix and fc layers
    state_dict_encoder = {}
    state_dict_projector = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
      if k.startswith('projection_head_imaging.'):
        state_dict_projector[k[len('projection_head_imaging.'):]] = state_dict[k]
      if k.startswith('model.backbone.'):
        state_dict_encoder[k[len('model.backbone.'):]] = state_dict[k]
      if k.startswith('model.head.'):
        state_dict_projector[k[len('model.head.'):]] = state_dict[k]

    log = self.backbone.load_state_dict(state_dict_encoder, strict=True)
    assert len(log.missing_keys) == 0

    if self.keep_projector:
      self.projection_head = SimCLRProjectionHead(pooled_dim, original_args['embedding_dim'], original_args['projection_dim'])

      log = self.projection_head.load_state_dict(state_dict_projector, strict=True)
      assert len(log.missing_keys) == 0

      pooled_dim = original_args['projection_dim']

      if self.vec2vec:
        modules = [
          nn.Linear(original_args['projection_dim'], original_args['projection_dim']),
          nn.ReLU(),
          nn.Linear(original_args['projection_dim'], original_args['projection_dim'])]

        self.vec2vec_head = nn.Sequential(*modules)

        checkpoint_v2v = torch.load(args.checkpoint_vec2vec)
        state_dict_v2v = checkpoint_v2v['state_dict']

        for k in list(state_dict_v2v.keys()):
          if k.startswith('net.'):
            state_dict_v2v[k[len('net.'):]] = state_dict_v2v[k]
          del state_dict_v2v[k]

        self.vec2vec_head.load_state_dict(state_dict_v2v, strict=True)

    # If we want more than just a linear classifier of the encodings
    if args.eval_classifier == 'mlp':
      self.head = nn.Sequential(OrderedDict([
        ('fc1',  nn.Linear(pooled_dim, int(pooled_dim/4))),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2',  nn.Linear(int(pooled_dim/4), int(pooled_dim/16))),
        ('relu2', nn.ReLU(inplace=True)),
        ('fc3',  nn.Linear(int(pooled_dim/16), args.num_classes))
      ]))
    else:
      self.head = nn.Linear(pooled_dim, args.num_classes)

    # Freeze if needed
    if args.finetune_strategy == 'frozen':
      for _, param in self.backbone.named_parameters():
        param.requires_grad = False
      parameters = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))
      assert len(parameters)==0
      if self.keep_projector:
        for _, param in self.projection_head.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.projection_head.parameters()))
        assert len(parameters)==0
        if self.vec2vec:
          for _, param in self.vec2vec_head.named_parameters():
            param.requires_grad = False
          parameters = list(filter(lambda p: p.requires_grad, self.vec2vec_head.parameters()))
          assert len(parameters)==0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.backbone(x).squeeze()
    if self.keep_projector:
      x = self.projection_head(x)
      if self.vec2vec:
        x = self.vec2vec_head(x)
    x = self.head(x)
    return x