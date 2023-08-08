import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
  """
  Simple linear classifier that is a single fully connected layer from input to class prediction.
  """
  def __init__(self, in_size: int, num_classes: int, init_type: str) -> None:
    super(LinearClassifier, self).__init__()
    self.model = nn.Linear(in_size, num_classes)
    self.init_type = init_type
    self.model.apply(self.init_weights)
    
  def init_weights(self, m, init_gain = 0.02) -> None:
    """
    Initializes weights according to desired strategy
    """
    if isinstance(m, nn.Linear):
      if self.init_type == 'normal':
        nn.init.normal_(m.weight.data, 0, 0.001)
      elif self.init_type == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=init_gain)
      elif self.init_type == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif self.init_type == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=init_gain)
      if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model(x)