import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


def gaussian_nll(mu, log_sigma, noise):
    NLL = torch.sum(log_sigma, 1) + torch.sum(((noise - mu) / (1e-8 + torch.exp(log_sigma))) ** 2, 1) / 2.
    return NLL.mean()


def schedule(p):
    return 2.0 / (1.0 + math.exp(- 10.0 * p)) - 1


def numpy_to_variable(x):
    return Variable(torch.from_numpy(x).cuda())


def log_sum_exp(logits, mask=None, inf=1e7):
    if mask is not None:
        logits = logits * mask - inf * (1.0 - mask)
        max_logits = logits.max(1)[0].unsqueeze(1)
        return ((logits - max_logits.expand_as(logits)).exp() * mask).sum(1).log().squeeze() + max_logits.squeeze()
    else:
        max_logits = logits.max(1)[0].unsqueeze(1)
        # numerical stability
        return ((logits - max_logits.expand_as(logits)).exp()).sum(1).log().squeeze() + max_logits.squeeze()


def log_sum_exp_0(logits):
    max_logits = logits.max()
    return (logits - max_logits.expand_as(logits)).exp().sum().log() + max_logits


def entropy(logits):
    probs = nn.functional.softmax(logits, dim=1)
    ent = (- probs * logits).sum(1).squeeze() + log_sum_exp(logits)
    return ent.mean()


def one_hot(logits, labels):
    mask = Variable(torch.zeros(logits.size(0), logits.size(1)).cuda())
    mask.data.scatter_(1, labels.data.view(-1, 1), 1)
    return mask


def grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
