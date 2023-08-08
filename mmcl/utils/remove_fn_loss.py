from typing import Tuple, List

import torch
from torch import nn

class RemoveFNLoss(torch.nn.Module):
  """
  Loss function for multimodal contrastive learning based off of the CLIP paper.
  
  Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
  similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
  Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal. 
  """
  def __init__(self, 
               temperature: float,
               cosine_similarity_matrix_path: str = None,
               threshold: float = 0.9) -> None:
    super(RemoveFNLoss, self).__init__()

    self.temperature = temperature
    self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    
    self.cosine_similarity_matrix = torch.load(cosine_similarity_matrix_path, map_location='cuda')
    self.cosine_similarity_matrix = torch.threshold(self.cosine_similarity_matrix, threshold, 0)
    self.cosine_similarity_matrix.fill_diagonal_(0)


  def forward(self, out0: torch.Tensor, out1: torch.Tensor, indices: List[int] = None) -> Tuple:
    # normalize the embedding onto the unit hypersphere
    out0 = nn.functional.normalize(out0, dim=1)
    out1 = nn.functional.normalize(out1, dim=1)
    labels = self.cosine_similarity_matrix[indices,:][:,indices]

    logits = torch.div(
            torch.cat([torch.matmul(out0, out1.T), #v1v2
                      torch.matmul(out1, out0.T)], #v2v1
                      dim=0),
            self.temperature)
    
    fn_mask = (labels==0)
    fn_mask = fn_mask.repeat(2,1)

    logits_mask = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    logits_mask = logits_mask.repeat(2,1)

    # compute log_prob
    exp_logits = torch.exp(logits)
    log_prob = logits[logits_mask] - torch.log((fn_mask * exp_logits).sum(1))

    loss = (-log_prob).mean()

    return loss, torch.matmul(out0, out1.T), torch.arange(len(out0), device=out0.device)