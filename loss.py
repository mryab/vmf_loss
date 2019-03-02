import torch.nn as nn
from torch.nn.functional import cosine_similarity


class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_voc, emb_dim):
        super(EmbeddingLoss, self).__init__()
        self.tgt_embedding = nn.Embedding(
            len(tgt_voc.vocab), emb_dim).from_pretrained(tgt_voc.vocab.vectors)
        #self.tgt_voc = tgt_voc
        self.pad_id = tgt_voc.vocab.stoi['<pad>']

    def forward(self, preds, target):
        return 0


class L2Loss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(L2Loss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return ((preds.transpose(1, 2) - target_embedded)
                ** 2).sum(dim=2).masked_select(mask).mean()


class CosineLoss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(CosineLoss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return 1 - cosine_similarity(preds.transpose(1, 2),
                                     target_embedded,
                                     dim=2,
                                     eps=1e-8).masked_select(mask).mean()
