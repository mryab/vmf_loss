import torch.nn as nn
import torch
from torch.nn.functional import cosine_similarity


class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_voc, emb_dim):
        super(EmbeddingLoss, self).__init__()
        self.tgt_embedding = nn.Embedding(len(tgt_voc.vocab),
                                          emb_dim).from_pretrained(tgt_voc.vocab.vectors)
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
        return 1 - cosine_similarity(preds.transpose(1, 2),
                                     target_embedded, dim=2, eps=1e-8)[mask].mean()


class MaxMarginLoss(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim):
        super(MaxMarginLoss, self).__init__(tgt_voc, emb_dim)

    def forward(self, preds, target):
        target_norm = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)  # batch x seq

        out_norm = normalize(preds.transpose(1, 2), p=2, dim=2)  # batch x seq x dim
        voc_norm = normalize(self.tgt_embedding.weight.data, p=2, dim=1)  # voc x dim

        gamma = 0.5

        cos_out_voc = out_norm.matmul(voc_norm.t())  # batch x seq x voc
        maxvalues, jmax = torch.max(
            cos_out_voc - target_norm.matmul(voc_norm.t()), dim=2, keepdim=True)  # batch x seq x 1

        cos_target = cos_out_voc.gather(2, target.unsqueeze(2)).squeeze()
        max_cos_voc = cos_out_voc.gather(2, jmax).squeeze()  # batch x seq
        diff = gamma + max_cos_voc - cos_target

        return diff.max(torch.zeros(1))[mask].mean()


class NLLvMFLossBase(EmbeddingLoss):
    def __init__(self, tgt_voc, emb_dim, reg_1=0, reg_2=1):
        super(NLLvMFLossBase, self).__init__(tgt_voc, emb_dim)
        self.reg_1 = reg_1
        self.reg_2 = reg_2
        self.emb_dim = emb_dim

    def forward(self, preds, target):
        target_norm = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)  # batch x seq

        loss = - self._cmk(self.emb_dim / 2. + 0.5, preds.norm(p=2, dim=2)
                                 ) - self.reg_2 * (target_norm * preds).sum(dim=2)
        if self.reg_1 > 0:
            loss = loss + self.reg_1 * preds.norm(p=2, dim=2)

        return loss[mask].mean()

    def _cmk(self, v, z):
        """
            v: scalar,
            z: Tensor of shape batch x seq
            return: Tensor of shape batch x seq

            """
        pass


class NLLvMFApproxPaper(NLLvMFLossBase):
    def __init__(self, tgt_voc, emb_dim, reg_1=0, reg_2=1):
        super(NLLvMFApproxPaper, self).__init__(tgt_voc, emb_dim, reg_1, reg_2)

    def _cmk(self, v, z):
        return torch.sqrt((v + 1)**2 + z**2) - (v - 1) * \
            torch.log(v - 1 + torch.sqrt((v + 1)**2 + z**2))


class NLLvMFApproxFixed(NLLvMFLossBase):
    def __init__(self, tgt_voc, emb_dim, reg_1=0, reg_2=1):
        super(NLLvMFLossPaperFixed, self).__init__(tgt_voc, emb_dim, reg_1, reg_2)

    def _cmk(self, v, z):
        return torch.sqrt((v - 1)**2 + z**2) - (v - 1) * \
            torch.log(v - 1 + torch.sqrt((v - 1)**2 + z**2))
