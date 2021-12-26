import math

import torch
from torch import nn
import numpy as np


def kl_divergence_loss(mu, sigma):

    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both).
    std = torch.exp(sigma / 2)
    p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q_zx = torch.distributions.Normal(mu, std)

    # Sample the latent code z.
    z = q_zx.rsample()

    # 2. get the probabilities from the equation
    log_q_zx = q_zx.log_prob(z)
    log_p_z = p_z.log_prob(z)

    # kl
    kl = log_q_zx - log_p_z
    kl = kl.sum(-1)

    return z, kl


def log_sum_exp(value, dim, keepdim=False):
    """ Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))

