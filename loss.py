import torch.nn as nn
from torch.nn.functional import cosine_similarity


class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_voc, emb_dim, loss):
        super(EmbeddingLoss, self).__init__()
        self.tgt_embedding = nn.Embedding(len(tgt_voc.vocab), emb_dim).from_pretrained(tgt_voc.vocab.vectors)
        self.loss = loss

    def forward(self, preds, target):
        return self.loss(preds, self.tgt_embedding(target))


def L2Loss(pred, target):
    return ((pred - self.tgt_embedding(target)) ** 2).mean(dim=0)


def CosineLoss(pred, target):
    return 1 - cosine_similarity(pred, self.tgt_embedding(target), dim=-1, eps=1e-8)
