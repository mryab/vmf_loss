import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity


class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_voc, emb_dim):
        super(EmbeddingLoss, self).__init__()
        self.tgt_embedding = nn.Embedding(len(tgt_voc.vocab), emb_dim).from_pretrained(tgt_voc.vocab.vectors)
        # self.tgt_voc = tgt_voc
        self.pad_id = tgt_voc.vocab.stoi[tgt_voc.pad_token]

    def forward(self, preds, target):
        return 0


class L2Loss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(L2Loss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return torch.norm((preds - target_embedded.transpose(1, 2)), p=2, dim=1)[mask].mean()


class CosineLoss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(CosineLoss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return 1 - cosine_similarity(preds.transpose(1, 2), target_embedded, dim=2, eps=1e-8)[mask].mean()


class MaxMarginLoss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(MaxMarginLoss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)

        target_norm = torch.nn.functional.normalize(target_embedded, p=2, dim=2)
        out_norm = torch.nn.functional.normalize(preds.transpose(1, 2), p=2, dim=2)
        voc_norm = torch.nn.functional.normalize(self.tgt_embedding.weights.data, p=2, dim=1)

        cos_ihat_j = out_norm.matmul(voc_norm.t())
        maxvalues, jmax = torch.max(cos_ihat_j - target_norm.matmul(voc_norm.t()), dim=-1)
