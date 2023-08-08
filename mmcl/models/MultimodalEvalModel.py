import torch
import torch.nn as nn
from collections import OrderedDict

from models.ResnetEvalModel import ResnetEvalModel
from models.ECGEvalModel import ECGEvalModel

class MultimodalEvalModel(nn.Module):
  """
  Evaluation model for imaging and tabular data.
  """
  def __init__(self, args) -> None:
    super(MultimodalEvalModel, self).__init__()

    self.imaging_model = ResnetEvalModel(args)
    self.ecg_model = ECGEvalModel(args)
    in_dim = 4096
    if args.eval_classifier == 'mlp':
      self.head = nn.Sequential(OrderedDict([
        ('fc1',  nn.Linear(in_dim, int(in_dim/4))),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2',  nn.Linear(int(in_dim/4), int(in_dim/16))),
        ('relu2', nn.ReLU(inplace=True)),
        ('fc3',  nn.Linear(int(in_dim/16), args.num_classes))
      ]))
    else:
      self.head = nn.Linear(in_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_im = self.imaging_model.backbone(x[0]).squeeze()
    x_ecg = self.ecg_model.forward_features(x[1]).squeeze()
    x = torch.cat((x_im, x_ecg), dim=1)
    x = self.head(x)
    return x