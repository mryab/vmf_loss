import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_voc, loss):
        super(EmbeddingLoss, self).__init__()
        self.tgt_embedding = nn.Embedding(len(tgt_voc.vocab), tgt_voc.vocab.vectors.dim).from_pretrained(tgt_voc.vocab.vectors)
        self.loss = loss
        
    def forward(self, preds, target):
        return self.loss(preds, self.tgt_embedding(target))

    
def L2Loss(pred, target):
    return ((pred - target) ** 2).mean(dim=0)

def CosineLoss(pred, target):
    return 1 - cosine_similarity(pred, target, dim=-1, eps=1e-8)