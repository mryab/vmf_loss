import numpy as np
import scipy.special
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity, normalize


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, tgt_field, args):
        super().__init__(ignore_index=tgt_field.vocab.stoi[tgt_field.pad_token])


class EmbeddingLoss(nn.Module):
    def __init__(self, tgt_field, args):
        super().__init__()
        self.tgt_embedding = nn.Embedding.from_pretrained(tgt_field.vocab.vectors)
        self.pad_id = tgt_field.vocab.stoi[tgt_field.pad_token]

    def forward(self, preds, target):
        raise NotImplementedError


class L2Loss(EmbeddingLoss):
    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return torch.norm((preds - target_embedded.transpose(1, 2)), p=2, dim=1)[
            mask
        ].mean()


class CosineLoss(EmbeddingLoss):
    def forward(self, preds, target):
        target_embedded = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)
        return (
            1
            - cosine_similarity(
                preds.transpose(1, 2), target_embedded, dim=2, eps=1e-8
            )[mask].mean()
        )


class MaxMarginLoss(EmbeddingLoss):
    def forward(self, preds, target):
        target_norm = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)  # batch x seq

        out_norm = normalize(preds.transpose(1, 2), p=2, dim=2)  # batch x seq x dim
        voc_norm = normalize(self.tgt_embedding.weight.data, p=2, dim=1)  # voc x dim

        gamma = 0.5

        cos_out_voc = out_norm.matmul(voc_norm.transpose(0, 1))  # batch x seq x voc
        maxvalues, jmax = torch.max(
            cos_out_voc - target_norm.matmul(voc_norm.transpose(0, 1)),
            dim=2,
            keepdim=True,
        )

        cos_target = cos_out_voc.gather(2, target.unsqueeze(2)).squeeze()
        max_cos_voc = cos_out_voc.gather(2, jmax).squeeze()  # batch x seq
        diff = gamma + max_cos_voc - cos_target

        return diff.max(torch.zeros(1).to(diff.device))[mask].mean()


class NLLvMFLossBase(EmbeddingLoss):
    def __init__(self, tgt_field, args):
        super().__init__(tgt_field, args)
        self.reg_1 = args.reg_1
        self.reg_2 = args.reg_2
        self.emb_dim = tgt_field.vocab.vectors.size(1)

    def forward(self, preds, target):
        target_norm = self.tgt_embedding(target)
        mask = target.ne(self.pad_id)  # batch x seq
        loss = (
            -self._logcmk(self.emb_dim, preds.norm(p=2, dim=1))
            - self.reg_2 * (target_norm * preds.transpose(1, 2)).sum(dim=2)
            + self.reg_1 * preds.norm(p=2, dim=1)
        )

        return loss[mask].mean()

    def _logcmk(self, v, z):
        """
        v: scalar,
        z: Tensor of shape batch x seq
        return: Tensor of shape batch x seq
        """
        raise NotImplementedError


class NLLvMFApproxPaper(NLLvMFLossBase):
    def _logcmk(self, m, z):
        v = m / 2.0 + 0.5
        return torch.sqrt((v + 1) ** 2 + z ** 2) - (v - 1) * torch.log(
            v - 1 + torch.sqrt((v + 1) ** 2 + z ** 2)
        )


class NLLvMFApproxFixed(NLLvMFLossBase):
    def _logcmk(self, m, z):
        v = m / 2.0 + 0.5
        return torch.sqrt((v - 1) ** 2 + z ** 2) - (v - 1) * torch.log(
            v - 1 + torch.sqrt((v - 1) ** 2 + z ** 2)
        )


class NLLvMF(NLLvMFLossBase):
    def _logcmk(self, m, z):
        return LogCMK.apply(m, z)


class LogCMK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k):
        ctx.save_for_backward(k)
        ctx.m = m
        k = k.double()
        ive = torch.from_numpy(scipy.special.ive(m / 2.0 - 1, k.cpu().numpy())).to(
            k.device
        )
        return (
            (m / 2.0 - 1) * torch.log(k) - torch.log(ive) - (m / 2) * np.log(2 * np.pi)
        ).float()

    @staticmethod
    def backward(ctx, grad_output):
        k = ctx.saved_tensors[0]
        m = ctx.m
        k = k.double().cpu().numpy()
        grads = -((scipy.special.ive(m / 2.0, k)) / (scipy.special.ive(m / 2.0 - 1, k)))
        return (
            None,
            grad_output * torch.from_numpy(grads).to(grad_output.device).float(),
        )


loss_registry = {
    "xent": CrossEntropyLoss,
    "l2": L2Loss,
    "cosine": CosineLoss,
    "maxmarg": MaxMarginLoss,
    "vmfapprox_paper": NLLvMFApproxPaper,
    "vmfapprox_fixed": NLLvMFApproxFixed,
    "vmf": NLLvMF,
}


def get_available_losses():
    return list(loss_registry.keys())


def get_loss(args, tgt_field):
    return loss_registry[args.loss](tgt_field, args)
