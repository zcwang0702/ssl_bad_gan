import torch
import torch.nn as nn


def d_criterion(pred, label):
    return nn.CrossEntropyLoss()(pred, label)


def gaussian_nll(mu, log_sigma, noise):
    NLL = torch.sum(log_sigma, 1) + torch.sum(((noise - mu) / (1e-8 + torch.exp(log_sigma))) ** 2, 1) / 2.
    return NLL.mean()


def log_sum_exp(logits, mask=None, inf=1e7):
    if mask is not None:
        logits = logits * mask - inf * (1.0 - mask)
        max_logits = logits.max(1)[0].unsqueeze(1)
        return ((logits - max_logits.expand_as(logits)).exp() * mask).sum(1).log().squeeze() + max_logits.squeeze()
    else:
        max_logits = logits.max(1)[0].unsqueeze(1)
        # numerical stability
        return ((logits - max_logits.expand_as(logits)).exp()).sum(1).log().squeeze() + max_logits.squeeze()


def entropy(logits):
    probs = nn.functional.softmax(logits, dim=1)
    ent = (- probs * logits).sum(1).squeeze() + log_sum_exp(logits)
    return ent.mean()
