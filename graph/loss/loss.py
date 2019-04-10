import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def discretized_mix_logistic_loss(x, l, sum_all=True):
    xs = x.size()  # (B,32,32,C)
    ls = l.size()  # (B,32,32,100)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)  # 10

    logit_probs = l[:, :, :, :nr_mix]  # size: [B, 32, 32, 3, nr_mix]
    # l = l[:, :, :, nr_mix:].contiguous().view(xs[0], xs[1], xs[2], xs[3], nr_mix * 3) # size: [B, 32, 32, 3, 3 * nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs[0], xs[1], xs[2], xs[3], -1)  # size: [B, 32, 32, C, 9 * nr_mix / C]

    # size: [B, 32, 32, C, nr_mix]
    means = l[:, :, :, :, :nr_mix]
    log_scales = F.threshold(l[:, :, :, :, nr_mix:2 * nr_mix], -7., -7.)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.unsqueeze(4).expand(xs[0], xs[1], xs[2], xs[3], nr_mix)  # size: [B, 32, 32, C, nr_mix]

    m1 = means[:, :, :, 0, :]
    m2 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m3 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    means = torch.cat([m1, m2, m3], 3)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    mask1 = (cdf_delta > 1e-5).float().detach()
    term1 = mask1 * torch.log(F.threshold(cdf_delta, 1e-12, 1e-12)) + (1. - mask1) * (log_pdf_mid - np.log(127.5))

    mask2 = (x > 0.999).float().detach()
    term2 = mask2 * log_one_minus_cdf_min + (1. - mask2) * term1

    mask3 = (x < -0.999).float().detach()
    term3 = mask3 * log_cdf_plus + (1. - mask3) * term2

    log_probs = term3.sum(3) + log_prob_from_logits(logit_probs)

    if not sum_all:
        return -log_sum_exp(log_probs).sum(1).sum(2).squeeze()
    else:
        return -log_sum_exp(log_probs).sum()


def discretized_mix_logistic_loss_c1(x, l, sum_all=True):
    xs = x.size()  # (B,32,32,1)
    ls = l.size()  # (B,32,32,100)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)

    logit_probs = l[:, :, :, :nr_mix]  # size: [B, 32, 32, nr_mix]
    # l = l[:, :, :, nr_mix:].contiguous().view(xs[0], xs[1], xs[2], xs[3], nr_mix * 3) # size: [B, 32, 32, 3, 3 * nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs[0], xs[1], xs[2], xs[3],
                                              nr_mix * 2)  # size: [B, 32, 32, 1, 2 * nr_mix]

    # size: [B, 32, 32, 1, nr_mix]
    means = l[:, :, :, :, :nr_mix]
    log_scales = F.threshold(l[:, :, :, :, nr_mix:2 * nr_mix], -7., -7.)
    # coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])

    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.unsqueeze(4).expand(xs[0], xs[1], xs[2], xs[3], nr_mix)  # size: [B, 32, 32, C, nr_mix]

    # m1 = means[:, :, :, 0, :]
    # m2 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    # m3 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    # means = torch.cat([m1, m2, m3], 3)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    mask1 = (cdf_delta > 1e-5).float().detach()
    term1 = mask1 * torch.log(F.threshold(cdf_delta, 1e-12, 1e-12)) + (1. - mask1) * (log_pdf_mid - np.log(127.5))

    mask2 = (x > 0.999).float().detach()
    term2 = mask2 * log_one_minus_cdf_min + (1. - mask2) * term1

    mask3 = (x < -0.999).float().detach()
    term3 = mask3 * log_cdf_plus + (1. - mask3) * term2

    log_probs = term3.sum(3) + log_prob_from_logits(logit_probs)

    if not sum_all:
        return -log_sum_exp(log_probs).sum(2).sum(1)
    else:
        return -log_sum_exp(log_probs).sum()


def log_prob_from_logits(logits):
    dim = logits.dim() - 1
    max_logits = logits.max(dim)[0].unsqueeze(dim).expand_as(logits)
    return logits - max_logits - (logits - max_logits).exp().sum(dim).unsqueeze(dim).log().expand_as(logits)
